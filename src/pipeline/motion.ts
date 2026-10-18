import { fileURLToPath } from "node:url";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

import type { RouteDef } from "@harness/template";
import type { ModelRef } from "@harness/template/types";
import type { CanonicalPart } from "@harness/template/llm/protocol";

import { routeForModel } from "../llm/routes.ts";
import { generateOnce } from "../llm/generate.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import { createFileTrajectoryWriter } from "../trajectory/writer.ts";
import { createStageEmitter, type StageEmitter } from "../trajectory/emitter.ts";
import { computeBBox, loadSTL, type STLMesh } from "../mesh/stl.ts";
import { writeOBJ } from "../mesh/obj.ts";
import {
  compilePartsInAssembly,
  extractAssemblyStatementsOf,
  listModuleInstances,
  listMotionParts,
} from "../scad/parts.ts";
import {
  createDefaultMotionPlan,
  normalizeMotionPlan,
  validateMotionPlan,
  type MotionAxis,
  type MotionCollisionApproximation,
  type MotionJointSpec,
  type MotionPlan,
} from "../motion/types.ts";
import {
  INCREMENTAL_MOTION_FILE,
  loadIncrementalMotionSidecar,
  type MeasuredBBox,
  type MotionJointKind,
} from "../motion/incremental.ts";
import { loadIncrementalAssemblySidecar } from "../motion/assembly.ts";
import { reconcileIncrementalAssembly, buildSeedAssemblyGraph, demoteDriftedAssembly } from "./assembly-seed.ts";
import { addStage, timeStage } from "./stage-timer.ts";
import { mapPool, COMPILE_CONCURRENCY } from "../util/pool.ts";
import type { AssemblyFixedEdge } from "./assembly-seed.ts";
import {
  buildSeedMotionPlan,
  demoteDriftedRecords,
  reconcileIncrementalMotion,
  seedBboxesFromRecords,
  trustedMeasuredAxis,
  type MatchedMotionRecord,
  type ReconciledIncrementalMotion,
} from "./motion-seed.ts";
import { meshToUsdMeshData, writeMotionUsda, type MotionLinkMesh } from "../motion/usda.ts";
import { writeMotionUrdf } from "../motion/urdf.ts";
import {
  findIsaacSim,
  runIsaacValidation,
  type IsaacValidationHints,
  type IsaacValidationReport,
} from "../motion/isaac.ts";
import { buildGeometricEvidence, type GeometricEvidence } from "../motion/geometry.ts";
import { renderPartsColorViews } from "../render/parts_color.ts";
import { splitScadToColoredParts } from "../render/parts_split.ts";
import { DEFAULT_VIEWS } from "../render/views.ts";

const PROCEDURA_ROOT = resolve(dirname(fileURLToPath(new URL(import.meta.url))), "..", "..");
const MOTION_PLAN_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "motion_plan_system.md");
const MOTION_AUTHOR_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "motion_author_system.md");
const MOTION_REFINE_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "motion_refine_system.md");
const MOTION_MAX_ATTEMPTS = 3;
const MAX_SCAD_CONTEXT_CHARS = 260_000;

export const DEFAULT_MOTION_MODEL = DEFAULT_MODEL;

export interface RunMotionExportOpts {
  outputDir: string;
  scadPath?: string;
  motionPlanPath?: string;
  /** When no sidecar is supplied, use a two-call LLM planner/author pipeline.
   *  Default true. Set false to use the legacy fixed-joint default plan. */
  useLlm?: boolean;
  /** Vision/reasoning model for motion planning. Default DEFAULT_MOTION_MODEL. */
  model?: string;
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
  /** Render per-part colour feedback before planning. Default true. */
  renderFeedback?: boolean;
  /** Run one post-author LLM repair pass using deterministic motion feedback.
   *  Default true for LLM-generated plans. */
  /** Post-author LLM repair pass. Default ON. It is the most expensive pass with
   *  the least visible effect, but across 186
   *  archived runs it changes joint topology in 26% of them, so it is NOT
   *  safe to drop by default on the strength of a wall-clock argument. */
  refine?: boolean;
  /** The second whole-plan authoring pass. Default ON — it changes topology in
   *  38% of 190 archived runs. `--no-author` exists to measure it, not to be a
   *  recommended configuration. */
  author?: boolean;
  /** Also export URDF (motion/urdf/robot.urdf + meshes). Default false. */
  exportUrdf?: boolean;
  /** Validate the exported asset headlessly in Isaac Sim. Default true when an
   *  Isaac install is found (PROCEDURA_ISAACSIM_PATH). */
  validate?: boolean;
  /** Simulation frames for the Isaac dynamic test. Default 120. */
  validateSteps?: number;
  /** One extra LLM refine round fed with the Isaac validation report when
   *  validation flags issues. Default true for LLM-generated plans. */
  simRefine?: boolean;
  /** Consume the incremental-draft motion sidecar (motion_incremental.json in
   *  the run dir) as planner priors + fallback seed. Default true; env kill
   *  switch PROCEDURA_INCREMENTAL_MOTION=0 also disables it. */
  incrementalMotion?: boolean;
  /** Isaac Sim install dir override. */
  isaacPath?: string;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  trajectoryPathOverride?: string;
}

export interface MotionValidationSummary {
  ran: boolean;
  /** Hard gate: schema audit + dynamic sim test + actuation sweep all pass. */
  ok?: boolean;
  /** Full report verdict including advisory asset-validation rules. */
  strictOk?: boolean;
  reportPath?: string;
  assetRuleErrors?: number;
  /** Per-joint drive sweep verdict (present when the actuation phase ran). */
  actuationOk?: boolean;
  /** No non-adjacent link pairs in contact at rest (advisory). */
  contactsOk?: boolean;
  /** Wheeled base moved under wheel drive (advisory). */
  mobilityOk?: boolean;
  framesCount?: number;
  videosCount?: number;
  errors?: string[];
  skippedReason?: string;
}

/** "incremental-seed" = deterministic seed built from the incremental-draft
 *  motion sidecar, used only as the fallback when the LLM planner fails or is
 *  disabled (the LLM path keeps "llm"). */
export type MotionPlanSource = "sidecar" | "llm" | "incremental-seed" | "default";

export interface MotionExportResult {
  ok: boolean;
  outputDir: string;
  usdaPath?: string;
  urdfPath?: string;
  planPath?: string;
  manifestPath?: string;
  planSource?: MotionPlanSource;
  planner?: MotionPlannerMetadata;
  validation?: MotionValidationSummary;
  simRefined?: boolean;
  linkCount: number;
  jointCount: number;
  warnings: string[];
  errors: string[];
  durationMs: number;
}

interface MotionManifestLink {
  name: string;
  parts: string[];
  stlPath: string;
  objPath?: string;
  pointCount: number;
  faceCount: number;
}

interface MotionPlannerMetadata {
  model?: string;
  contextPath?: string;
  designBriefPath?: string;
  renderDir?: string;
  renderViews?: string[];
  draftPlanPath?: string;
  plannerResponsePath?: string;
  authorResponsePath?: string;
  refineFeedbackPath?: string;
  finalFeedbackPath?: string;
  refinedPlanPath?: string;
  refinerResponsePath?: string;
  simRefineFeedbackPath?: string;
  simRefineResponsePath?: string;
  simRefinedPlanPath?: string;
  thinkingPath?: string;
  notes?: string[];
}

interface MotionLlmContext {
  objectText: string;
  parts: MotionPartContext[];
  designBrief: MotionDesignBrief;
  renderLegend: string;
  scadContext: string;
  images: MotionImageContext[];
}

interface MotionPlanResolution {
  plan: MotionPlan;
  source: MotionPlanSource;
  warnings: string[];
  planner?: MotionPlannerMetadata;
  /** Present for LLM-resolved plans; reused by the sim-feedback refine round. */
  llmContext?: MotionLlmContext;
}

interface MotionPartContext {
  name: string;
  description?: string;
  assembly?: string[];
  bbox?: {
    min: [number, number, number];
    max: [number, number, number];
    size: [number, number, number];
  };
  center?: [number, number, number];
  diagonal?: number;
  volume?: number;
  semanticTags?: string[];
  motionRole?: string;
  motionHint?: {
    preferredJointTypes: MotionJointSpec["type"][];
    likelyAxes: MotionAxis[];
    roleReason: string;
  };
}

interface MotionImageContext {
  label: string;
  path: string;
  b64: string;
}

interface MotionRefineFeedback {
  validator: {
    ok: boolean;
    errors: string[];
    warnings: string[];
  };
  coverage: {
    totalParts: number;
    assignedParts: number;
    missingParts: string[];
    duplicateParts: Array<{ part: string; links: string[] }>;
    unknownParts: Array<{ part: string; link: string }>;
  };
  topology: {
    rootLink?: string;
    linkCount: number;
    jointCount: number;
    nonFixedJointCount: number;
    orphanLinks: string[];
    multipleParentLinks: Array<{ link: string; parents: string[] }>;
    cycles: string[][];
  };
  links: Array<{
    name: string;
    parts: string[];
    bbox?: MotionPartContext["bbox"];
  }>;
  joints: Array<{
    name: string;
    type: string;
    parent?: string;
    child: string;
    axis?: string;
    anchor?: [number, number, number];
    anchorInsideParent?: boolean;
    anchorInsideChild?: boolean;
    distanceToParentBBox?: number;
    distanceToChildBBox?: number;
    issues: string[];
  }>;
  semanticExpectations?: {
    mechanismHypotheses: MotionDesignBrief["mechanismHypotheses"];
    anchorCandidates: MotionDesignBrief["anchorCandidates"];
  };
  /** Headless Isaac Sim validation results, attached for the sim-feedback refine round. */
  isaacValidation?: IsaacValidationReport | { errors: string[] };
  likelyIssues: string[];
}

interface MotionDesignBrief {
  summary: {
    partCount: number;
    bboxAvailable: number;
    likelyMobileBase: boolean;
    likelyStaticFixture: boolean;
  };
  coordinateSystem: {
    x: string;
    y: string;
    z: string;
    units: string;
    jointFrameNote: string;
  };
  partRoles: Array<{
    part: string;
    role: string;
    semanticTags: string[];
    center?: [number, number, number];
    size?: [number, number, number];
    motionHint: MotionPartContext["motionHint"];
  }>;
  symmetryGroups: Array<{
    kind: "left_right" | "front_rear";
    negative: string;
    positive: string;
    mirroredAxis: MotionAxis;
    centerDelta: [number, number, number];
  }>;
  proximityPairs: Array<{
    a: string;
    b: string;
    bboxGap: number;
    centerDistance: number;
    relation: string;
  }>;
  anchorCandidates: Array<{
    part: string;
    role: string;
    preferredJointType: MotionJointSpec["type"];
    axis?: MotionAxis;
    anchor: [number, number, number];
    confidence: "high" | "medium" | "low";
    reason: string;
  }>;
  mechanismHypotheses: Array<{
    name: string;
    evidenceParts: string[];
    recommendedTopology: string;
    recommendedJointTypes: MotionJointSpec["type"][];
    notes: string[];
  }>;
  /** Deterministic mesh-derived evidence: rotational-symmetry axes and near-contact strips. */
  geometricEvidence?: GeometricEvidence;
  /** Per-part articulation intent declared during incremental generation
   *  (categorical; the declared role already lives in partRoles), with the
   *  build-time MEASURED axis when trusted (high confidence). */
  incrementalPriors?: Array<{
    part: string;
    moving: boolean;
    jointType?: MotionJointKind;
    parent?: string;
    axis?: MotionAxis;
    measuredAxis: MotionAxis | null;
    axisAgrees: boolean | null;
  }>;
  /** Build-time sidecar warnings plus consume-time staleness warnings. */
  incrementalWarnings?: string[];
  plannerChecklist: string[];
}

export async function runMotionExport(opts: RunMotionExportOpts): Promise<MotionExportResult> {
  const t0 = Date.now();
  const outputDir = resolve(opts.outputDir);
  const scadPath = opts.scadPath ?? join(outputDir, "final.scad");
  const motionDir = join(outputDir, "motion");
  const linksDir = join(motionDir, "links");
  const warnings: string[] = [];
  const errors: string[] = [];
  mkdirSync(linksDir, { recursive: true });

  if (!existsSync(scadPath)) {
    return fail(`motion export: SCAD file not found: ${scadPath}`);
  }

  const scadCode = readFileSync(scadPath, "utf8");
  // Instance-expanded part universe: modules placed more than once appear as
  // <module>__i<k> instances so reused/mirrored parts can become separate links.
  const partNames = listMotionParts(scadCode);
  if (partNames.length === 0) {
    return fail(`motion export: no top-level SCAD modules found in ${scadPath}`);
  }

  // Motion sidecar left by the incremental draft loop (in the RUN dir, not
  // motion/). Reconciled records become planner priors and a deterministic
  // seed plan; a missing/stale sidecar degrades to today's exact behavior.
  // Gated by the incrementalMotion opt and the PROCEDURA_INCREMENTAL_MOTION=0
  // env kill switch (same switch the draft side honors).
  let incremental: ReconciledIncrementalMotion | null = null;
  const incrementalEnabled = (opts.incrementalMotion ?? true)
    && process.env["PROCEDURA_INCREMENTAL_MOTION"] !== "0";
  if (incrementalEnabled) {
    const incrementalSidecar = await loadIncrementalMotionSidecar(outputDir);
    if (incrementalSidecar !== null) {
      incremental = reconcileIncrementalMotion(incrementalSidecar, partNames);
      warnings.push(...incremental.warnings);
      opts.log?.(`  motion: incremental sidecar found (${incremental.matched.length} matched, ${incremental.stale.length} stale record(s))`);
    }
  } else if (existsSync(join(outputDir, INCREMENTAL_MOTION_FILE))) {
    opts.log?.("  motion: incremental sidecar present but disabled (incrementalMotion=false or PROCEDURA_INCREMENTAL_MOTION=0); ignoring it");
  }

  // Assembly-aware handoff (Slice 4): reconcile the STATIC-mate sidecar against
  // the final part universe and write the deterministic fixed-joint skeleton to
  // motion/assembly_seed.json — the explicit static edges (child fixed to its
  // mating partner, with a measured contact anchor) that the motion seed
  // otherwise collapses into one root link. Gated by the same incrementalMotion
  // opt + kill switch; a missing sidecar is a silent no-op. Consumed by URDF/BOM
  // downstream and available as planner context (fixed anchors).
  // The ENTIRE optional assembly handoff is best-effort: a missing/malformed
  // sidecar or any reconcile/seed error must never abort the motion export.
  let assemblyFixedEdges: AssemblyFixedEdge[] = [];
  try {
    if (incrementalEnabled) {
      const assemblySidecar = await loadIncrementalAssemblySidecar(outputDir);
      if (assemblySidecar !== null) {
        const reconciled = reconcileIncrementalAssembly(assemblySidecar, partNames);
        // Drift gate: a whole-model refine may have moved geometry since the
        // records were measured. Demote records whose part bbox drifted before
        // they become verified edges (mirrors the motion drift gate).
        const driftIds = reconciled.matched
          .filter((m) => m.record.measured?.partBBox && m.instanceIds.length === 1)
          .map((m) => m.instanceIds[0]!);
        if (driftIds.length > 0) {
          const currentBboxes = await computeCurrentInstanceBboxes(
            scadCode, driftIds, join(motionDir, "_assembly_drift"), opts.log,
          );
          const demoted = demoteDriftedAssembly(reconciled.matched, currentBboxes);
          reconciled.matched = demoted.matched;
          reconciled.warnings.push(...demoted.warnings);
        }
        const graph = buildSeedAssemblyGraph(reconciled.matched);
        assemblyFixedEdges = graph.edges; // planner injection filters mates===true
        warnings.push(...reconciled.warnings, ...graph.warnings);
        mkdirSync(motionDir, { recursive: true });
        writeFileSync(
          join(motionDir, "assembly_seed.json"),
          `${JSON.stringify({
            source: "incremental-assembly", edges: graph.edges,
            stale: reconciled.stale.length, warnings: [...reconciled.warnings, ...graph.warnings],
          }, null, 2)}\n`,
          "utf8",
        );
        const registered = graph.edges.filter((e) => e.mates === true).length;
        opts.log?.(
          `  motion: assembly sidecar found (${reconciled.matched.length} matched, ` +
          `${reconciled.stale.length} stale) — ${graph.edges.length} fixed edge(s), ` +
          `${registered} verified-mating → assembly_seed.json`,
        );
      }
    }
  } catch (e) {
    warnings.push(`assembly handoff skipped (non-fatal): ${(e as Error).message.slice(0, 160)}`);
  }

  let failPlanSource: MotionPlanSource = "default";
  let planResolution: MotionPlanResolution;
  try {
    planResolution = await resolveMotionPlan({
      outputDir,
      motionDir,
      scadPath,
      scadCode,
      motionPlanPath: opts.motionPlanPath,
      partNames,
      incremental,
      assemblyEdges: assemblyFixedEdges,
      fixedBase: opts.fixedBase,
      defaultCollision: opts.defaultCollision,
      useLlm: opts.useLlm,
      model: opts.model,
      renderFeedback: opts.renderFeedback,
      refine: opts.refine,
      author: opts.author,
      log: opts.log,
      trajectorySink: opts.trajectorySink,
      trajectoryPathOverride: opts.trajectoryPathOverride,
    });
  } catch (e) {
    return fail((e as Error).message);
  }
  failPlanSource = planResolution.source;
  let plan = planResolution.plan;
  warnings.push(...planResolution.warnings);
  const validation = validateMotionPlan(plan, partNames);
  warnings.push(...validation.warnings);
  if (!validation.ok) {
    return {
      ok: false,
      outputDir,
      planSource: planResolution.source,
      ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
      linkCount: plan.links.length,
      jointCount: plan.joints.length,
      warnings,
      errors: validation.errors,
      durationMs: Date.now() - t0,
    };
  }

  const planPath = join(motionDir, opts.motionPlanPath ? "motion_plan.resolved.json" : "motion_plan.generated.json");
  writeFileSync(planPath, JSON.stringify(plan, null, 2) + "\n", "utf8");

  rmSync(linksDir, { recursive: true, force: true });
  mkdirSync(linksDir, { recursive: true });

  const usdaPath = join(motionDir, "final_motion.usda");
  // Compiled link geometry survives the sim-refine re-export when a link keeps
  // the same name and part grouping.
  const meshCache = new Map<string, MotionLinkMesh>();

  interface ExportedArtifacts {
    linkMeshes: MotionLinkMesh[];
    manifestLinks: MotionManifestLink[];
    urdfPath?: string;
  }

  const compileAndWrite = async (current: MotionPlan): Promise<ExportedArtifacts | null> => {
    const linkMeshes: MotionLinkMesh[] = [];
    const manifestLinks: MotionManifestLink[] = [];
    // Link compiles are independent OpenSCAD runs — pool them. Two passes,
    // because the cache is keyed on (safe name, part set): dedupe FIRST so two
    // links sharing a key compile once, then fold in plan order so the manifest,
    // the error list and the log are byte-identical to the sequential version.
    type LinkSpec = MotionPlan["links"][number];
    interface LinkJob { link: LinkSpec; parts: string[]; safeName: string }
    const specs = current.links.map((link): [string, LinkJob] => {
      const parts = link.parts && link.parts.length > 0 ? link.parts : [link.name];
      const safeName = safePathName(link.name);
      return [`${safeName}::${[...parts].sort().join("|")}`, { link, parts, safeName }];
    });
    const pending = new Map<string, LinkJob>();
    for (const [cacheKey, job] of specs) {
      if (!meshCache.has(cacheKey) && !pending.has(cacheKey)) pending.set(cacheKey, job);
    }
    const built = await timeStage("motion.links", () =>
      mapPool([...pending], COMPILE_CONCURRENCY, async ([cacheKey, job]) => {
        const linkDir = join(linksDir, job.safeName);
        const stlPath = await compilePartsInAssembly(scadCode, job.parts, linkDir);
        if (stlPath === null) return { cacheKey, job, failed: true as const };
        const stableStlPath = join(linkDir, `${job.safeName}.stl`);
        const stableObjPath = join(linkDir, `${job.safeName}.obj`);
        copyFileSync(stlPath, stableStlPath);
        const mesh = loadSTL(stableStlPath);
        // compileScad no longer emits output.obj by default, so write the
        // per-link OBJ ourselves from the mesh we already loaded (the motion
        // manifest and downstream URDF meshes reference it).
        let objError: string | undefined;
        try { writeOBJ(stableObjPath, mesh); }
        catch (e) { objError = (e as Error).message; }
        return {
          cacheKey,
          job,
          compiled: {
            link: job.link,
            mesh: meshToUsdMeshData(mesh),
            stlPath: stableStlPath,
            ...(existsSync(stableObjPath) ? { objPath: stableObjPath } : {}),
          } satisfies MotionLinkMesh,
          ...(objError !== undefined ? { objError } : {}),
        };
      }));
    for (const b of built) {
      opts.log?.(`  motion link: ${b.job.link.name} (${b.job.parts.join(", ")})`);
      if ("failed" in b) {
        errors.push(`failed to compile link ${b.job.link.name} from parts: ${b.job.parts.join(", ")}`);
        continue;
      }
      if (b.objError !== undefined) {
        opts.log?.(`  motion link ${b.job.link.name}: OBJ write failed: ${b.objError}`);
      }
      meshCache.set(b.cacheKey, b.compiled);
    }

    for (const [cacheKey, job] of specs) {
      const compiled = meshCache.get(cacheKey);
      if (compiled === undefined) continue; // its compile failed; errors already has it
      linkMeshes.push({ ...compiled, link: job.link });
      manifestLinks.push({
        name: job.link.name,
        parts: job.parts,
        stlPath: compiled.stlPath!,
        ...(compiled.objPath !== undefined ? { objPath: compiled.objPath } : {}),
        pointCount: compiled.mesh.points.length,
        faceCount: compiled.mesh.faceVertexCounts.length,
      });
    }
    if (errors.length > 0) return null;

    // The sim-refine round re-authors the same file; dedupe so repeated
    // authoring warnings do not accumulate across rounds.
    for (const authored of writeMotionUsda({ path: usdaPath, plan: current, linkMeshes, warnings })) {
      if (!warnings.includes(authored)) warnings.push(authored);
    }

    let urdfPath: string | undefined;
    if (opts.exportUrdf) {
      try {
        const urdf = writeMotionUrdf({
          urdfDir: join(motionDir, "urdf"),
          plan: current,
          linkMeshes,
          name: "robot",
        });
        urdfPath = urdf.urdfPath;
        warnings.push(...urdf.warnings.map((w) => `urdf: ${w}`));
      } catch (err) {
        warnings.push(`urdf export failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    return { linkMeshes, manifestLinks, ...(urdfPath !== undefined ? { urdfPath } : {}) };
  };

  const validateArtifacts = async (urdfPath: string | undefined): Promise<{
    summary: MotionValidationSummary;
    report: IsaacValidationReport | null;
    needsRefine: boolean;
  }> => {
    const isaacPath = opts.isaacPath ?? findIsaacSim() ?? undefined;
    if (isaacPath === undefined) {
      const reason = "Isaac Sim not found (set PROCEDURA_ISAACSIM_PATH)";
      warnings.push(`isaac validation skipped: ${reason}`);
      return { summary: { ran: false, skippedReason: reason }, report: null, needsRefine: false };
    }
    // Plan-derived hints unlock the actuation sweep / contact scan / mobility
    // phases; the frames dir collects RGB captures of the swept poses.
    const hintsPath = join(motionDir, "isaac_hints.json");
    writeFileSync(hintsPath, JSON.stringify(buildIsaacHints(plan), null, 2) + "\n", "utf8");
    const framesDir = join(motionDir, "isaac_frames");
    rmSync(framesDir, { recursive: true, force: true });
    const videosDir = join(motionDir, "isaac_videos");
    rmSync(videosDir, { recursive: true, force: true });
    const run = await timeStage("motion.isaac", () => runIsaacValidation({
      usdaPath,
      ...(urdfPath !== undefined ? { urdfPath } : {}),
      reportPath: join(motionDir, "isaac_validation.json"),
      steps: opts.validateSteps ?? 120,
      isaacPath,
      hintsPath,
      framesDir,
      videosDir,
      ...(opts.log !== undefined ? { log: opts.log } : {}),
    }));
    const report = run.report;
    const simulation = report?.phases.simulation;
    const actuation = report?.phases.actuation;
    const hardOk = report !== null
      && !run.timedOut
      && run.exitCode === 0
      && report.phases.schemaAudit?.ok === true
      && (simulation?.ok === true || simulation?.skipped === true)
      && (actuation === undefined || actuation.skipped === true || actuation.ok === true);
    const assetRuleErrors = (report?.phases.assetRules?.issues ?? [])
      .filter((issue) => issue.severity.toLowerCase().includes("error")).length;
    if (!hardOk) {
      warnings.push(...run.errors.map((e) => `isaac validation: ${e}`));
    } else if (report !== null && report.ok !== true) {
      warnings.push(
        `isaac validation passed the hard gate (schema + simulation + actuation) but reported ` +
        `${assetRuleErrors} advisory asset-rule error(s); see ${run.reportPath}`,
      );
    }
    const contacts = report?.phases.contacts;
    const mobility = report?.phases.mobility;
    if (contacts?.skipped !== true && contacts?.ok === false) {
      warnings.push(`isaac validation: ${contacts.nonAdjacentRestCount ?? 0} non-adjacent link pair(s) in contact at rest`);
    }
    if (mobility?.skipped !== true && mobility?.ok === false) {
      warnings.push(`isaac validation: wheeled base barely moved under wheel drive (${(mobility.baseDisplacement ?? 0).toFixed(1)} units)`);
    }
    return {
      summary: {
        ran: true,
        ok: hardOk,
        strictOk: report?.ok === true,
        reportPath: run.reportPath,
        assetRuleErrors,
        ...(actuation !== undefined && actuation.skipped !== true ? { actuationOk: actuation.ok === true } : {}),
        ...(contacts !== undefined && contacts.skipped !== true ? { contactsOk: contacts.ok === true } : {}),
        ...(mobility !== undefined && mobility.skipped !== true ? { mobilityOk: mobility.ok === true } : {}),
        ...(report?.frames !== undefined ? { framesCount: report.frames.length } : {}),
        ...(report?.videos !== undefined ? { videosCount: report.videos.length } : {}),
        ...(run.errors.length > 0 ? { errors: run.errors } : {}),
      },
      report,
      needsRefine: !hardOk || report?.ok !== true,
    };
  };

  // Compile/author/validate can throw (OpenSCAD, filesystem, Isaac spawn);
  // surface those as a failed result instead of crashing the caller.
  try {
    let artifacts = await compileAndWrite(plan);
    if (artifacts === null) {
      return {
        ok: false,
        outputDir,
        planPath,
        planSource: planResolution.source,
        ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
        linkCount: 0,
        jointCount: plan.joints.length,
        warnings,
        errors,
        durationMs: Date.now() - t0,
      };
    }

    let validationSummary: MotionValidationSummary = { ran: false, skippedReason: "disabled" };
    let simRefined = false;
    if (opts.validate ?? true) {
      let checked = await validateArtifacts(artifacts.urdfPath);
      validationSummary = checked.summary;

      const canSimRefine = (opts.simRefine ?? true)
        && planResolution.source === "llm"
        && planResolution.llmContext !== undefined;
      if (checked.summary.ran && checked.needsRefine && canSimRefine) {
        opts.log?.("  motion: Isaac validation flagged issues; running sim-feedback refine");
        const refined = await simRefineMotionPlan({
          outputDir,
          motionDir,
          ...(opts.model !== undefined ? { model: opts.model } : {}),
          context: planResolution.llmContext!,
          currentPlan: plan,
          partNames,
          isaacReport: checked.report,
          isaacErrors: checked.summary.errors ?? [],
          simFrames: checked.report?.frames ?? [],
          ...(opts.fixedBase !== undefined ? { fixedBase: opts.fixedBase } : {}),
          ...(opts.defaultCollision !== undefined ? { defaultCollision: opts.defaultCollision } : {}),
          ...(opts.log !== undefined ? { log: opts.log } : {}),
          ...(opts.trajectorySink !== undefined ? { trajectorySink: opts.trajectorySink } : {}),
          ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
        });
        warnings.push(...refined.warnings);
        if (refined.plan) {
          plan = refined.plan;
          const reexported = await compileAndWrite(plan);
          // The canonical plan file is only replaced once the re-export succeeded,
          // so it always matches the artifacts on disk.
          if (reexported !== null) {
            writeFileSync(planPath, JSON.stringify(plan, null, 2) + "\n", "utf8");
          }
          if (reexported === null) {
            return {
              ok: false,
              outputDir,
              planPath,
              planSource: planResolution.source,
              ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
              simRefined: true,
              linkCount: 0,
              jointCount: plan.joints.length,
              warnings,
              errors,
              durationMs: Date.now() - t0,
            };
          }
          artifacts = reexported;
          simRefined = true;
          checked = await validateArtifacts(artifacts.urdfPath);
          validationSummary = checked.summary;
        }
      }
    }

    const manifestPath = join(motionDir, "manifest.json");
    writeFileSync(
      manifestPath,
      JSON.stringify({
        usdaPath,
        ...(artifacts.urdfPath !== undefined ? { urdfPath: artifacts.urdfPath } : {}),
        planPath,
        sourceScadPath: scadPath,
        metersPerUnit: plan.metersPerUnit,
        fixedBase: plan.fixedBase,
        planSource: planResolution.source,
        ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
        validation: validationSummary,
        simRefined,
        ...(incremental !== null ? {
          incrementalPriors: incremental.matched.length,
          incrementalStale: incremental.stale.length,
          incrementalDrifted: incremental.drifted?.length ?? 0,
          seedUsedAsFallback: planResolution.source === "incremental-seed",
        } : {}),
        links: artifacts.manifestLinks,
        joints: plan.joints.map((joint) => ({
          name: joint.name,
          type: joint.type,
          parent: joint.parent ?? null,
          child: joint.child,
        })),
        warnings,
      }, null, 2) + "\n",
      "utf8",
    );

    return {
      ok: true,
      outputDir,
      usdaPath,
      ...(artifacts.urdfPath !== undefined ? { urdfPath: artifacts.urdfPath } : {}),
      planPath,
      manifestPath,
      planSource: planResolution.source,
      ...(planResolution.planner !== undefined ? { planner: planResolution.planner } : {}),
      validation: validationSummary,
      simRefined,
      linkCount: artifacts.linkMeshes.length,
      jointCount: plan.joints.length,
      warnings,
      errors,
      durationMs: Date.now() - t0,
    };
  } catch (e) {
    return fail(`motion export failed: ${(e as Error).message}`);
  }

  function fail(message: string): MotionExportResult {
    return {
      ok: false,
      outputDir,
      planSource: failPlanSource,
      linkCount: 0,
      jointCount: 0,
      warnings,
      errors: [...errors.filter((e) => e !== message), message],
      durationMs: Date.now() - t0,
    };
  }
}

async function resolveMotionPlan(args: {
  outputDir: string;
  motionDir: string;
  scadPath: string;
  scadCode: string;
  motionPlanPath?: string;
  partNames: string[];
  incremental?: ReconciledIncrementalMotion | null;
  /** (#10) Verified static-mate edges → injected as fixed anchorCandidates. */
  assemblyEdges?: AssemblyFixedEdge[];
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
  useLlm?: boolean;
  model?: string;
  renderFeedback?: boolean;
  refine?: boolean;
  author?: boolean;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  trajectoryPathOverride?: string;
}): Promise<MotionPlanResolution> {
  if (args.motionPlanPath) {
    return {
      plan: loadSidecarPlan({ ...args, motionPlanPath: args.motionPlanPath }),
      source: "sidecar",
      warnings: [],
    };
  }

  // Deterministic seed from the incremental-build sidecar: a prompt draft for
  // the LLM planner, and the fallback plan replacing the all-fixed default.
  const seedWarnings: string[] = [];
  const incremental = args.incremental ?? null;
  if (incremental !== null) {
    // Drift gate: refine may have moved geometry since the records were
    // measured. Compare against freshly compiled instance bboxes and demote
    // drifted records to categorical priors before they feed the rails/seed.
    const measuredIds = incremental.matched
      .filter((m) => m.record.measured?.bbox !== undefined && m.instanceIds.length === 1)
      .map((m) => m.instanceIds[0]!);
    if (measuredIds.length > 0) {
      const currentBboxes = await computeCurrentInstanceBboxes(
        args.scadCode,
        measuredIds,
        join(args.motionDir, "_incremental_drift"),
        args.log,
      );
      const demotion = demoteDriftedRecords(incremental.matched, currentBboxes);
      incremental.matched = demotion.matched;
      incremental.drifted = demotion.drifted;
      incremental.warnings.push(...demotion.warnings);
      seedWarnings.push(...demotion.warnings);
    }
  }
  const seed = incremental !== null && incremental.matched.length > 0
    ? buildSeedMotionPlan(incremental.matched, args.partNames, seedBboxesFromRecords(incremental.matched), seedWarnings)
    : null;
  if (incremental !== null) {
    writeIncrementalPriorsFile(args.motionDir, incremental, seed);
    if (seed !== null) {
      args.log?.(`  motion: incremental seed plan built (${seed.links.length} links, ${seed.joints.length} joints)`);
    }
  }

  if (args.useLlm !== false) {
    const llm = await buildLlmMotionPlan({ ...args, incrementalSeed: seed });
    if (llm) return { ...llm, warnings: [...seedWarnings, ...llm.warnings] };
  }

  if (seed !== null) {
    return {
      plan: applyPlanOverrides(seed, args),
      source: "incremental-seed",
      warnings: [
        ...seedWarnings,
        args.useLlm === false
          ? "used incremental-seed motion plan (LLM planner disabled)"
          : "LLM motion planning did not produce a valid plan; used the incremental-seed plan built from build-time measurements",
      ],
    };
  }

  return {
    plan: createDefaultPlanWithOverrides(args),
    source: "default",
    warnings: [
      ...seedWarnings,
      ...(args.useLlm === false
        ? []
        : ["LLM motion planning did not produce a valid plan; used fixed-joint default plan"]),
    ],
  };
}

/** Debug artifact: the reconciled incremental priors + seed actually consumed. */
function writeIncrementalPriorsFile(
  motionDir: string,
  incremental: ReconciledIncrementalMotion,
  seed: MotionPlan | null,
): void {
  try {
    mkdirSync(motionDir, { recursive: true });
    writeFileSync(
      join(motionDir, "motion_incremental_priors.json"),
      JSON.stringify({
        matched: incremental.matched,
        stale: incremental.stale,
        drifted: incremental.drifted ?? [],
        warnings: incremental.warnings,
        sidecarWarnings: incremental.sidecarWarnings,
        seed,
      }, null, 2) + "\n",
      "utf8",
    );
  } catch {
    // Debug artifact only — never fatal.
  }
}

/** Freshly compiled world bboxes for the given instance ids (drift check).
 *  Best-effort: ids that fail to compile are simply absent from the map. */
async function computeCurrentInstanceBboxes(
  scadCode: string,
  instanceIds: string[],
  outDir: string,
  log?: (line: string) => void,
): Promise<Map<string, MeasuredBBox>> {
  // One OpenSCAD compile per instance, pooled like every other per-module
  // compile in the codebase. Results are folded back in INPUT order so the
  // map contents and the failure log stay independent of completion order.
  const compiled = await mapPool(instanceIds, COMPILE_CONCURRENCY, async (id) => {
    try {
      const stlPath = await compilePartsInAssembly(scadCode, [id], join(outDir, safePathName(id)));
      if (stlPath === null) return null;
      return { id, bbox: computeBBox(loadSTL(stlPath)) };
    } catch (err) {
      return { id, error: err instanceof Error ? err.message : String(err) };
    }
  });
  const out = new Map<string, MeasuredBBox>();
  for (const r of compiled) {
    if (r === null) continue;
    if ("error" in r) { log?.(`  motion: drift-check compile failed for ${r.id}: ${r.error}`); continue; }
    out.set(r.id, r.bbox);
  }
  return out;
}

/** Appended to the planner system prompt ONLY when a reconciled incremental
 *  sidecar with usable records exists, so standalone/non-incremental runs get
 *  a byte-identical system prompt. */
const MOTION_PLAN_INCREMENTAL_ADDENDUM =
  "This model was built incrementally: you additionally receive a `DRAFT MOTION PLAN` section and `incrementalPriors` inside the design brief, " +
  "produced by the incremental builder that modeled each part. Joint types, parents, and roles there were declared while modeling, and anchors/axes " +
  "flagged as measured come from deterministic mesh analysis of the as-built geometry. Treat them as strong priors — CORRECT and complete the draft " +
  "plan instead of starting from scratch, prefer its measured anchors and axes, refine its bbox-center anchors (those are low-confidence placeholders), " +
  "and do not drop a declared joint without cause; override only where the renders or SCAD source clearly disagree. Never reference parts that are not " +
  "in AVAILABLE TOP-LEVEL SCAD PARTS.";

/** Exported for the incremental-handoff smoke test (pure string builder). */
export function buildPlannerSystemPrompt(incremental: ReconciledIncrementalMotion | null): string {
  const base = readFileSync(MOTION_PLAN_SYSTEM_PATH, "utf8");
  if (incremental === null || incremental.matched.length === 0) return base;
  return `${base}\n${MOTION_PLAN_INCREMENTAL_ADDENDUM}\n`;
}

function loadSidecarPlan(args: {
  motionPlanPath: string;
  partNames: string[];
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
}): MotionPlan {
  let raw: unknown;
  if (!existsSync(args.motionPlanPath)) {
    throw new Error(`motion export: motion plan not found: ${args.motionPlanPath}`);
  }
  raw = JSON.parse(readFileSync(args.motionPlanPath, "utf8")) as unknown;
  if (!hasLinksArray(raw)) {
    throw new Error(`motion export: motion plan must contain a links array`);
  }
  const plan = normalizeMotionPlan(raw, args.partNames);
  return applyPlanOverrides(plan, args);
}

function createDefaultPlanWithOverrides(args: {
  partNames: string[];
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
}): MotionPlan {
  return createDefaultMotionPlan({
    partNames: args.partNames,
    fixedBase: args.fixedBase,
    defaultCollision: args.defaultCollision,
  });
}

/**
 * Plan-derived joint hints for the Isaac actuation/contacts/mobility phases.
 * Mimic followers are excluded from driven/wheel sets — the coupling actuates
 * them and the validator checks them via mimicJoints instead.
 */
/** Something the base ROLLS ON. A thumbwheel, a steering wheel and a film
 *  roller all contain "wheel"/"roller" and none of them is locomotion. */
const LOCOMOTION_WHEEL = /wheel|tire|tyre|roller/i;
const NOT_LOCOMOTION = new RegExp(
  // `[\s_-]*` because these arrive as identifiers: `steering_wheel`, not
  // `steering wheel`. A pattern that only allowed a space silently matched
  // nothing.
  "(?:thumb|hand|fly|pin|gear|steering|control)[\\s_-]*wheel"
  + "|dial|knob|crank|spool|reel|winder|propeller|rotor|turret|fan|impeller",
  "i",
);

export function buildIsaacHints(plan: MotionPlan): IsaacValidationHints {
  const mimicFollowers = new Set(plan.joints.filter((joint) => joint.mimic !== undefined).map((joint) => joint.name));
  const linkByName = new Map(plan.links.map((link) => [link.name, link]));
  const wheelLike = (joint: MotionJointSpec): boolean => {
    const link = linkByName.get(joint.child);
    const text = `${joint.child} ${(link?.parts ?? []).join(" ")}`;
    return LOCOMOTION_WHEEL.test(text) && !NOT_LOCOMOTION.test(text);
  };
  const drivenJoints = plan.joints
    .filter((joint) => (joint.type === "revolute" || joint.type === "prismatic")
      && joint.drive !== undefined
      && !mimicFollowers.has(joint.name))
    .map((joint) => ({
      name: joint.name,
      mode: ((joint.drive?.stiffness ?? 0) > 0 ? "position" : "velocity") as "position" | "velocity",
      lower: joint.limit?.lower ?? joint.limit?.low ?? null,
      upper: joint.limit?.upper ?? joint.limit?.high ?? null,
      targetVelocity: joint.drive?.targetVelocity ?? null,
    }));
  // Only NAMED locomotion wheels. This used to also accept any velocity-driven
  // revolute (`stiffness === 0`), which is how a TLR camera's focus knob, film
  // advance knob and spool knob became wheels: one archived run spent 179.6s
  // driving a camera across a ground plane, failed, and warned that the wheeled
  // base barely moved. Quadcopters drove their propellers; a submarine drove its
  // screw. Across 18 archived mobility runs that clause produced 6 failures, all
  // false, and 242 of 449 seconds.
  //
  // It also over-drove the genuine vehicles: rover runs spun their suspension
  // rockers and bogies, an assault buggy its steering wheel, a biplane its
  // propeller — none of which is how the thing rolls. Narrowing to named wheels
  // makes those tests cleaner as well as the non-vehicles free.
  //
  // The cost is a plan whose wheels are named so obliquely that nothing matches
  // loses the test. Mobility is ADVISORY and never gates the verdict, so that
  // costs a signal, never a pass.
  const wheelJoints = plan.joints
    .filter((joint) => joint.type === "revolute"
      && !mimicFollowers.has(joint.name)
      && wheelLike(joint))
    .map((joint) => joint.name);
  const mimicJoints = plan.joints
    .filter((joint) => joint.mimic !== undefined)
    .map((joint) => ({
      follower: joint.name,
      reference: joint.mimic!.referenceJoint,
      gearing: joint.mimic!.gearing ?? 1,
      offset: joint.mimic!.offset ?? 0,
    }));
  return {
    rootLink: plan.rootLink ?? plan.links[0]?.name ?? "root",
    fixedBase: plan.fixedBase ?? true,
    drivenJoints,
    wheelJoints,
    mimicJoints,
  };
}

function applyPlanOverrides(
  plan: MotionPlan,
  args: { fixedBase?: boolean; defaultCollision?: MotionCollisionApproximation },
): MotionPlan {
  // normalizeMotionPlan has already baked the plan's default collision into
  // every link, so the override must also rewrite links carrying that baked
  // default (explicit per-link choices differing from the default are kept).
  const bakedDefault = plan.defaultCollision ?? "convexHull";
  return {
    ...plan,
    ...(args.fixedBase !== undefined ? { fixedBase: args.fixedBase } : {}),
    ...(args.defaultCollision !== undefined ? { defaultCollision: args.defaultCollision } : {}),
    links: plan.links.map((link) => ({
      ...link,
      ...(args.defaultCollision !== undefined && (link.collision === undefined || link.collision === bakedDefault)
        ? { collision: args.defaultCollision }
        : {}),
    })),
  };
}

async function buildLlmMotionPlan(args: {
  outputDir: string;
  motionDir: string;
  scadPath: string;
  scadCode: string;
  partNames: string[];
  incremental?: ReconciledIncrementalMotion | null;
  assemblyEdges?: AssemblyFixedEdge[];
  incrementalSeed?: MotionPlan | null;
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
  model?: string;
  renderFeedback?: boolean;
  refine?: boolean;
  author?: boolean;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  trajectoryPathOverride?: string;
}): Promise<MotionPlanResolution | null> {
  const log = args.log ?? (() => undefined);
  const warnings: string[] = [];
  const metadata: MotionPlannerMetadata = {
    model: args.model ?? DEFAULT_MOTION_MODEL,
    notes: [],
  };

  // The plan is authored up to three times in a row by three prompts that ask
  // for the SAME JSON shape from the same inputs, which is 80% of the stage's
  // wall clock. Both later passes look inert on any single model — on a 7-link
  // buggy neither changed joint topology across six runs, and three of five
  // refines were exact identity maps. Across 186/190 ARCHIVED runs spanning five
  // model families, though, refine changes topology in 26% and author in 38%.
  // So both stay ON; the flags exist to ablate them, not as a fast preset.
  const refineEnabled = args.refine !== false;
  const authorEnabled = args.author !== false;
  const requiredPrompts = [
    MOTION_PLAN_SYSTEM_PATH,
    ...(authorEnabled ? [MOTION_AUTHOR_SYSTEM_PATH] : []),
    ...(refineEnabled ? [MOTION_REFINE_SYSTEM_PATH] : []),
  ];
  const missingPrompts = requiredPrompts.filter((p) => !existsSync(p));
  if (missingPrompts.length > 0) {
    warnings.push(`motion LLM prompts are missing: ${missingPrompts.join(", ")}`);
    return null;
  }

  mkdirSync(args.motionDir, { recursive: true });
  const context = await timeStage("motion.context",
    () => buildMotionContext(args, warnings, metadata));
  const contextPath = join(args.motionDir, "motion_llm_context.json");
  writeFileSync(contextPath, JSON.stringify({
    objectText: context.objectText,
    partCount: context.parts.length,
    parts: context.parts,
    designBriefPath: metadata.designBriefPath,
    designBrief: context.designBrief,
    images: context.images.map((img) => ({ label: img.label, path: img.path })),
    renderLegend: context.renderLegend,
    scadPath: args.scadPath,
  }, null, 2) + "\n", "utf8");
  metadata.contextPath = contextPath;

  const modelKey = args.model ?? DEFAULT_MOTION_MODEL;
  const modelRef: ModelRef = resolveModel(modelKey);
  const route = routeForModel(modelKey);

  const trajectoryDir = join(args.outputDir, "_trajectory");
  mkdirSync(trajectoryDir, { recursive: true });
  const localWriter = args.trajectorySink
    ? null
    : createFileTrajectoryWriter(trajectoryDir, `motion-${Date.now().toString(36)}`);
  const sink = args.trajectorySink ?? localWriter!.sink;

  // No harness. This stage never calls a tool: it makes one-shot generations
  // and writes trajectory events. See src/trajectory/emitter.ts.
  const sessionId = `sess_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  const runId = `run_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  const emitter = createStageEmitter({
    sink, sessionId, workspaceDir: args.outputDir, runId,
    source: "motion", provider: modelRef.providerId, modelId: modelRef.modelId,
  });
  emitter.emit("run.started", { sessionId, runId, title: `motion: ${basename(args.outputDir)}` });

  try {
    const planSystem = buildPlannerSystemPrompt(args.incremental ?? null);
    const authorSystem = authorEnabled ? readFileSync(MOTION_AUTHOR_SYSTEM_PATH, "utf8") : "";
    const refineSystem = refineEnabled ? readFileSync(MOTION_REFINE_SYSTEM_PATH, "utf8") : "";

    const plannerUser = buildPlannerUserPrompt(context, args);
    const planner = await generateMotionJson({
      emitter,
      route,
      modelRef,
      label: "motion-plan",
      systemPrompt: planSystem,
      userText: plannerUser,
      images: context.images,
      log,
    });
    const plannerResponsePath = join(args.motionDir, "motion_plan_llm_response.txt");
    writeFileSync(plannerResponsePath, planner.raw, "utf8");
    metadata.plannerResponsePath = plannerResponsePath;

    const draftPlan = motionPlanFromPayload(planner.parsed, args.partNames);
    if (!draftPlan) {
      warnings.push("motion planner LLM did not return a plan with a links array");
      return null;
    }
    const draftPlanOverridden = applyPlanOverrides(draftPlan, args);
    const draftPlanPath = join(args.motionDir, "motion_plan.llm_draft.json");
    writeFileSync(draftPlanPath, JSON.stringify(draftPlanOverridden, null, 2) + "\n", "utf8");
    metadata.draftPlanPath = draftPlanPath;

    const draftValidation = validateMotionPlan(draftPlanOverridden, args.partNames);

    let authorReasoning = "";
    let finalPlanOverridden: MotionPlan | null = null;
    let finalValidation: ReturnType<typeof validateMotionPlan> | null = null;
    if (authorEnabled) {
      const authorUser = buildAuthorUserPrompt(context, args, draftPlanOverridden, draftValidation);
      const author = await generateMotionJson({
        emitter,
        route,
        modelRef,
        label: "motion-author",
        systemPrompt: authorSystem,
        userText: authorUser,
        images: context.images,
        log,
      });
      const authorResponsePath = join(args.motionDir, "motion_author_llm_response.txt");
      writeFileSync(authorResponsePath, author.raw, "utf8");
      metadata.authorResponsePath = authorResponsePath;
      authorReasoning = author.reasoning;

      const finalPlan = motionPlanFromPayload(author.parsed, args.partNames);
      finalPlanOverridden = finalPlan ? applyPlanOverrides(finalPlan, args) : null;
      finalValidation = finalPlanOverridden ? validateMotionPlan(finalPlanOverridden, args.partNames) : null;
    }

    let selectedPlan = finalPlanOverridden ?? draftPlanOverridden;
    let selectedValidation = finalValidation ?? draftValidation;
    let refinerReasoning = "";

    if (refineEnabled) {
      const feedback = buildMotionRefineFeedback(selectedPlan, selectedValidation, context, args.partNames);
      const feedbackPath = join(args.motionDir, "motion_refine_feedback.json");
      writeFileSync(feedbackPath, JSON.stringify(feedback, null, 2) + "\n", "utf8");
      metadata.refineFeedbackPath = feedbackPath;

      const refinerUser = buildRefineUserPrompt(context, args, selectedPlan, feedback);
      const refiner = await generateMotionJson({
        emitter,
        route,
        modelRef,
        label: "motion-refine",
        systemPrompt: refineSystem,
        userText: refinerUser,
        images: context.images,
        log,
      });
      const refinerResponsePath = join(args.motionDir, "motion_refine_llm_response.txt");
      writeFileSync(refinerResponsePath, refiner.raw, "utf8");
      metadata.refinerResponsePath = refinerResponsePath;
      refinerReasoning = refiner.reasoning;

      const refinedPlan = motionPlanFromPayload(refiner.parsed, args.partNames);
      const refinedPlanOverridden = refinedPlan ? applyPlanOverrides(refinedPlan, args) : null;
      const refinedValidation = refinedPlanOverridden ? validateMotionPlan(refinedPlanOverridden, args.partNames) : null;
      if (refinedPlanOverridden) {
        const refinedPlanPath = join(args.motionDir, "motion_plan.refined.json");
        writeFileSync(refinedPlanPath, JSON.stringify(refinedPlanOverridden, null, 2) + "\n", "utf8");
        metadata.refinedPlanPath = refinedPlanPath;
      }
      if (refinedPlanOverridden && refinedValidation?.ok) {
        selectedPlan = refinedPlanOverridden;
        selectedValidation = refinedValidation;
        metadata.notes?.push("one-step motion refine accepted");
      } else if (refinedValidation && !refinedValidation.ok) {
        warnings.push(`motion refiner plan failed validation: ${refinedValidation.errors.join("; ")}`);
      } else {
        warnings.push("motion refiner LLM did not return a plan with a links array");
      }
    }

    const thinking = [
      planner.reasoning ? `=== motion-plan ===\n${planner.reasoning}` : "",
      authorReasoning ? `=== motion-author ===\n${authorReasoning}` : "",
      refinerReasoning ? `=== motion-refine ===\n${refinerReasoning}` : "",
    ].filter(Boolean).join("\n\n");
    if (thinking) {
      const thinkingPath = join(args.motionDir, "motion_thinking.txt");
      writeFileSync(thinkingPath, thinking, "utf8");
      metadata.thinkingPath = thinkingPath;
    }

    if (selectedValidation.ok) {
      const finalFeedback = buildMotionRefineFeedback(selectedPlan, selectedValidation, context, args.partNames);
      const finalFeedbackPath = join(args.motionDir, "motion_final_feedback.json");
      writeFileSync(finalFeedbackPath, JSON.stringify(finalFeedback, null, 2) + "\n", "utf8");
      metadata.finalFeedbackPath = finalFeedbackPath;
      emitter.emit("motion.author.finished", {
        sessionId,
        links: selectedPlan.links.length,
        joints: selectedPlan.joints.length,
      });
      return {
        plan: selectedPlan,
        source: "llm",
        warnings: [
          ...warnings,
          ...draftValidation.warnings.map((w) => `draft plan: ${w}`),
          ...(finalValidation ? finalValidation.warnings.map((w) => `author plan: ${w}`) : []),
          ...selectedValidation.warnings,
        ],
        planner: metadata,
        llmContext: context,
      };
    }

    if (finalValidation && !finalValidation.ok) {
      warnings.push(`motion author plan failed validation: ${finalValidation.errors.join("; ")}`);
    } else {
      warnings.push("motion author LLM did not return a plan with a links array");
    }

    if (draftValidation.ok) {
      warnings.push("using validated draft articulation plan because final author plan was invalid");
      const fallbackFeedback = buildMotionRefineFeedback(draftPlanOverridden, draftValidation, context, args.partNames);
      const finalFeedbackPath = join(args.motionDir, "motion_final_feedback.json");
      writeFileSync(finalFeedbackPath, JSON.stringify(fallbackFeedback, null, 2) + "\n", "utf8");
      metadata.finalFeedbackPath = finalFeedbackPath;
      return {
        plan: draftPlanOverridden,
        source: "llm",
        warnings: [...warnings, ...draftValidation.warnings],
        planner: metadata,
        llmContext: context,
      };
    }

    warnings.push(`motion planner draft failed validation: ${draftValidation.errors.join("; ")}`);
    return null;
  } finally {
    emitter.emit("run.finished", { sessionId, runId, reason: "stop" });
    if (localWriter) await localWriter.close();
  }
}

/**
 * One extra refine call fed with the Isaac Sim validation report. Reuses the
 * planner's context (design brief, part facts, renders) so the LLM sees the
 * same evidence plus the simulation findings.
 */
async function simRefineMotionPlan(args: {
  outputDir: string;
  motionDir: string;
  model?: string;
  context: MotionLlmContext;
  currentPlan: MotionPlan;
  partNames: string[];
  isaacReport: IsaacValidationReport | null;
  isaacErrors: string[];
  /** RGB captures of swept poses from the Isaac run; attached as vision context. */
  simFrames?: Array<{ path: string; label: string }>;
  fixedBase?: boolean;
  defaultCollision?: MotionCollisionApproximation;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  planner?: MotionPlannerMetadata;
}): Promise<{ plan: MotionPlan | null; warnings: string[] }> {
  const log = args.log ?? (() => undefined);
  const warnings: string[] = [];
  if (!existsSync(MOTION_REFINE_SYSTEM_PATH)) {
    return { plan: null, warnings: [`sim refine skipped: missing prompt ${MOTION_REFINE_SYSTEM_PATH}`] };
  }

  const validation = validateMotionPlan(args.currentPlan, args.partNames);
  const feedback = buildMotionRefineFeedback(args.currentPlan, validation, args.context, args.partNames);
  feedback.isaacValidation = args.isaacReport ?? { errors: args.isaacErrors };
  const feedbackPath = join(args.motionDir, "motion_sim_refine_feedback.json");
  writeFileSync(feedbackPath, JSON.stringify(feedback, null, 2) + "\n", "utf8");
  if (args.planner) args.planner.simRefineFeedbackPath = feedbackPath;

  const modelKey = args.model ?? DEFAULT_MOTION_MODEL;
  const modelRef: ModelRef = resolveModel(modelKey);
  const route = routeForModel(modelKey);

  const trajectoryDir = join(args.outputDir, "_trajectory");
  mkdirSync(trajectoryDir, { recursive: true });
  const localWriter = args.trajectorySink
    ? null
    : createFileTrajectoryWriter(trajectoryDir, `motion-simrefine-${Date.now().toString(36)}`);
  const sink = args.trajectorySink ?? localWriter!.sink;

  const sessionId = `sess_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  const runId = `run_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  const emitter = createStageEmitter({
    sink, sessionId, workspaceDir: args.outputDir, runId,
    source: "motion-sim-refine", provider: modelRef.providerId, modelId: modelRef.modelId,
  });
  emitter.emit("run.started", { sessionId, runId, title: `motion-sim-refine: ${basename(args.outputDir)}` });

  try {
    const refineSystem = readFileSync(MOTION_REFINE_SYSTEM_PATH, "utf8");
    const refinerUser = buildRefineUserPrompt(args.context, args, args.currentPlan, feedback);
    // Attach the captured simulation frames so the critic can SEE wrong axes,
    // anchors, and limit poses instead of inferring them from numbers alone.
    const frameImages: MotionImageContext[] = [];
    for (const frame of (args.simFrames ?? []).slice(0, 8)) {
      try {
        frameImages.push({
          label: `Isaac simulation frame — ${frame.label}:`,
          path: frame.path,
          b64: readFileSync(frame.path).toString("base64"),
        });
      } catch {
        // Missing frame files are non-fatal; the numeric report still flows.
      }
    }
    const refiner = await generateMotionJson({
      emitter,
      route,
      modelRef,
      label: "motion-sim-refine",
      systemPrompt: refineSystem,
      userText: refinerUser,
      images: [...args.context.images, ...frameImages],
      log,
    });
    const responsePath = join(args.motionDir, "motion_sim_refine_llm_response.txt");
    writeFileSync(responsePath, refiner.raw, "utf8");
    if (args.planner) args.planner.simRefineResponsePath = responsePath;

    const refinedPlan = motionPlanFromPayload(refiner.parsed, args.partNames);
    if (!refinedPlan) {
      warnings.push("sim refine: LLM did not return a plan with a links array");
      return { plan: null, warnings };
    }
    const overridden = applyPlanOverrides(refinedPlan, args);
    const refinedValidation = validateMotionPlan(overridden, args.partNames);
    const refinedPlanPath = join(args.motionDir, "motion_plan.sim_refined.json");
    writeFileSync(refinedPlanPath, JSON.stringify(overridden, null, 2) + "\n", "utf8");
    if (args.planner) args.planner.simRefinedPlanPath = refinedPlanPath;
    if (!refinedValidation.ok) {
      warnings.push(`sim refine: plan failed validation: ${refinedValidation.errors.join("; ")}`);
      return { plan: null, warnings };
    }
    warnings.push(...refinedValidation.warnings.map((w) => `sim-refined plan: ${w}`));
    args.planner?.notes?.push("sim-feedback refine accepted");
    return { plan: overridden, warnings };
  } finally {
    emitter.emit("run.finished", { sessionId, runId, reason: "stop" });
    if (localWriter) await localWriter.close();
  }
}

/** Exported for the incremental-handoff smoke test (no LLM calls inside). */
export async function buildMotionContext(
  args: {
    outputDir: string;
    motionDir: string;
    scadPath: string;
    scadCode: string;
    partNames: string[];
    incremental?: ReconciledIncrementalMotion | null;
    assemblyEdges?: AssemblyFixedEdge[];
    renderFeedback?: boolean;
    log?: (line: string) => void;
  },
  warnings: string[],
  metadata: MotionPlannerMetadata,
): Promise<MotionLlmContext> {
  const log = args.log ?? (() => undefined);
  const objectText = loadObjectText(args.outputDir);
  const descs = loadPartDescriptions(args.outputDir);
  const bboxByPart = new Map<string, MotionPartContext["bbox"]>();
  const meshByPart = new Map<string, STLMesh>();
  let renderLegend = "";
  const images: MotionImageContext[] = [];

  // Multi-placement module instances: compile each selected placement on its
  // own so instance parts get real bboxes/meshes for planning and geometry.
  const instances = listModuleInstances(args.scadCode).filter((inst) => inst.instanceId !== inst.module);
  const moduleByInstance = new Map(instances.map((inst) => [inst.instanceId, inst.module]));
  const statementByInstance = new Map(instances.map((inst) => [inst.instanceId, inst.statement]));
  // Pooled: these are independent OpenSCAD compiles, one per placement. The
  // fold below runs in INPUT order so map contents and warning text do not
  // depend on which compile finished first.
  const wanted = instances.filter((inst) => args.partNames.includes(inst.instanceId));
  const instanceMeshes = await timeStage("motion.instances", () =>
    mapPool(wanted, COMPILE_CONCURRENCY, async (inst) => {
      const instDir = join(args.motionDir, "_instances", safePathName(inst.instanceId));
      try {
        const stlPath = await compilePartsInAssembly(args.scadCode, [inst.instanceId], instDir);
        if (stlPath === null) return { id: inst.instanceId, empty: true as const };
        return { id: inst.instanceId, mesh: loadSTL(stlPath) };
      } catch (err) {
        return { id: inst.instanceId, error: err instanceof Error ? err.message : String(err) };
      }
    }));
  for (const r of instanceMeshes) {
    if ("empty" in r) { warnings.push(`motion context: instance ${r.id} produced no geometry`); continue; }
    if ("error" in r) { warnings.push(`motion context: failed to compile instance ${r.id}: ${r.error}`); continue; }
    meshByPart.set(r.id, r.mesh);
    const bbox = computeBBox(r.mesh);
    bboxByPart.set(r.id, { min: bbox.min, max: bbox.max, size: bbox.size });
  }

  const refImage = join(args.outputDir, "image.png");
  if (existsSync(refImage)) {
    images.push({
      label: "Reference image of the intended object:",
      path: refImage,
      b64: readFileSync(refImage).toString("base64"),
    });
  }

  if (args.renderFeedback ?? true) {
    const renderDir = join(args.motionDir, "render_feedback");
    metadata.renderDir = renderDir;
    log("  motion: rendering per-part color feedback for LLM planner");
    const split = await timeStage("motion.split", () => splitScadToColoredParts({
      scadCode: args.scadCode,
      outDir: renderDir,
      log: (line) => log(`  ${line}`),
    }));
    if (split.ok) {
      renderLegend = split.legend;
      for (const part of split.parts) {
        try {
          const mesh = loadSTL(part.stl);
          if (!meshByPart.has(part.name)) meshByPart.set(part.name, mesh);
          const bbox = computeBBox(mesh);
          if (!bboxByPart.has(part.name)) {
            bboxByPart.set(part.name, {
              min: bbox.min,
              max: bbox.max,
              size: bbox.size,
            });
          }
        } catch {
          warnings.push(`motion context: failed to compute bbox for ${part.name}`);
        }
      }
      const render = await renderPartsColorViews({
        scadPath: args.scadPath,
        outDir: renderDir,
        preSplit: { meshSpecs: split.meshSpecs, legend: split.legend },
        views: DEFAULT_VIEWS,
        edges: true,
        log: (line) => log(`  ${line}`),
      });
      if (render.ok) {
        metadata.renderViews = render.views.map((v) => v.view);
        for (const view of render.views) {
          images.push({
            label: `Per-part color render feedback — ${view.view} view:`,
            path: view.path,
            b64: readFileSync(view.path).toString("base64"),
          });
        }
      } else {
        warnings.push(`motion context: render feedback failed: ${render.error}`);
      }
    } else {
      warnings.push(`motion context: per-part split failed: ${split.error}`);
    }
  }

  const rawParts: MotionPartContext[] = args.partNames.map((name) => {
    const instanceStatement = statementByInstance.get(name);
    const assembly = instanceStatement
      ? [compactWhitespace(instanceStatement, 900)]
      : extractAssemblyStatementsOf(args.scadCode, name)
          .map((s) => compactWhitespace(s, 900))
          .slice(0, 3);
    const description = descs.get(name) ?? descs.get(moduleByInstance.get(name) ?? "");
    return {
      name,
      ...(description ? { description } : {}),
      ...(assembly.length > 0 ? { assembly } : {}),
      ...(bboxByPart.get(name) ? { bbox: bboxByPart.get(name)! } : {}),
    };
  });
  const parts = enrichMotionParts(rawParts);
  // Incremental-build priors override the name-based heuristics BEFORE the
  // design brief snapshots partRoles, anchor candidates, and mechanism
  // hypotheses, so every rail sees the declared state, not the guessed one.
  if (args.incremental && args.incremental.matched.length > 0) {
    applyIncrementalPartOverrides(parts, args.incremental);
  }
  const designBrief = buildMotionDesignBrief({ objectText, parts });

  // Mesh-derived evidence: rotational-symmetry axes (likely spin/hinge axes)
  // and elongated near-contact strips (likely hinge lines), folded into the
  // anchor candidates the planner already consumes.
  try {
    const meshParts = args.partNames
      .filter((name) => meshByPart.has(name))
      .map((name) => ({ name, mesh: meshByPart.get(name)! }));
    if (meshParts.length > 0) {
      const evidence = buildGeometricEvidence(
        meshParts,
        designBrief.proximityPairs.map((pair) => ({ a: pair.a, b: pair.b })),
      );
      designBrief.geometricEvidence = evidence;
      for (const axis of evidence.symmetryAxes) {
        if (axis.confidence !== "high" || axis.degenerate) continue;
        designBrief.anchorCandidates.push({
          part: axis.part,
          role: "rotational_symmetry_axis",
          preferredJointType: "revolute",
          ...(axis.snappedAxis !== null ? { axis: axis.snappedAxis } : {}),
          anchor: axis.axisPoint,
          confidence: "high",
          reason: `mesh is rotationally symmetric about this axis (score ${axis.symmetryScore.toFixed(2)}); likely spin/hinge axis`,
        });
      }
      for (const region of evidence.contactRegions) {
        if (region.principalDir === null || region.elongation < 2) continue;
        designBrief.anchorCandidates.push({
          part: region.partB,
          role: "contact_strip",
          preferredJointType: "revolute",
          ...(region.snappedAxis !== null ? { axis: region.snappedAxis } : {}),
          anchor: region.anchor,
          confidence: "high",
          reason: `elongated near-contact strip with ${region.partA} (elongation ${region.elongation.toFixed(1)}); hinge lines run along such strips`,
        });
      }
    }
  } catch (err) {
    warnings.push(`motion context: geometric evidence failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  // Build-time MEASURED anchors plus the compact priors/warnings sections join
  // the brief after the heuristic/geometric candidates. Declared-only
  // (unmeasured) metadata never becomes a coordinate.
  if (args.incremental && args.incremental.matched.length > 0) {
    appendIncrementalBriefSections(designBrief, args.incremental);
  }
  // (#10) Verified static mates → fixed anchorCandidates, so the planner sees
  // the fixed-joint skeleton instead of collapsing statics into one root.
  if (args.assemblyEdges && args.assemblyEdges.length > 0) {
    appendIncrementalAssemblyBriefSections(designBrief, args.assemblyEdges);
  }

  const designBriefPath = join(args.motionDir, "motion_design_brief.json");
  writeFileSync(designBriefPath, JSON.stringify(designBrief, null, 2) + "\n", "utf8");
  metadata.designBriefPath = designBriefPath;

  return {
    objectText,
    parts,
    designBrief,
    renderLegend,
    scadContext: scadContext(args.scadCode),
    images,
  };
}

/**
 * Override the name-based heuristic guesses (motionRole/motionHint) on parts
 * the incremental builder declared. Runs BEFORE buildMotionDesignBrief so
 * partRoles, heuristic anchor candidates, and mechanism hypotheses all derive
 * from the declared state. Measured axes are used only at high confidence.
 */
function applyIncrementalPartOverrides(
  parts: MotionPartContext[],
  incremental: ReconciledIncrementalMotion,
): void {
  const matchedByInstance = new Map<string, MatchedMotionRecord>();
  for (const m of incremental.matched) {
    for (const id of m.instanceIds) matchedByInstance.set(id, m);
  }

  for (const part of parts) {
    const m = matchedByInstance.get(part.name);
    if (m === undefined) continue;
    const declared = m.record.declared;
    const measuredAxis = trustedMeasuredAxis(m.record);
    const role = declared.role ?? declared.jointType;
    if (role !== undefined) part.motionRole = role;
    part.motionHint = {
      preferredJointTypes: [declared.jointType ?? (declared.moving ? "revolute" : "fixed")],
      likelyAxes: [measuredAxis ?? declared.axis].filter((a): a is MotionAxis => a !== null && a !== undefined),
      roleReason: `Declared by the incremental builder${measuredAxis !== null ? "; axis measured from as-built geometry" : ""}`,
    };
  }
}

/**
 * Append the incremental sections to the design brief AFTER the heuristic and
 * geometric candidates: build-time MEASURED anchors as high-confidence anchor
 * candidates, the compact incrementalPriors table (role omitted — partRoles
 * already carries the overridden role), and the combined warnings.
 */
/** (#10) Inject VERIFIED static mates as `fixed` anchorCandidates so the motion
 *  planner is told which parts are rigidly attached to which (at a measured
 *  anchor), instead of the seed collapsing all statics into one root link. Only
 *  edges with mates===true + a resolved parent + a measured anchor are used
 *  (mates=false/null are excluded); this enriches the planner's context and is
 *  never a bypass of the planner. */
function appendIncrementalAssemblyBriefSections(
  designBrief: MotionDesignBrief,
  edges: AssemblyFixedEdge[],
): void {
  for (const e of edges) {
    if (e.mates !== true || !e.parent || !e.anchor) continue;
    designBrief.anchorCandidates.push({
      part: e.child,
      role: "incremental_static_mate",
      preferredJointType: "fixed",
      anchor: roundVec3([e.anchor[0], e.anchor[1], e.anchor[2]]),
      confidence: "high",
      reason: `verified static ${e.mate ?? "mate"} to ${e.parent}, measured from the as-built assembly — a fixed joint`,
    });
  }
}

function appendIncrementalBriefSections(
  designBrief: MotionDesignBrief,
  incremental: ReconciledIncrementalMotion,
): void {
  const axisNotes: string[] = [];
  for (const m of incremental.matched) {
    const declared = m.record.declared;
    const measured = m.record.measured;
    const sym = measured?.symmetryAxis;
    // Below-trust-bar measured axes are demoted to the declared axis; note it
    // only when the two actually disagree.
    if (sym !== undefined && sym.confidence !== "high" && sym.snappedAxis !== null
      && declared.axis !== undefined && sym.snappedAxis !== declared.axis) {
      axisNotes.push(
        `${m.record.placedName}: measured axis ${sym.snappedAxis} (confidence ${sym.confidence}) ` +
        `disagrees with declared ${declared.axis}; using declared (below trust bar)`,
      );
    }
    if (!declared.moving) continue;
    const anchor = measured?.parentContact?.anchor ?? measured?.symmetryAxis?.axisPoint;
    // Measured anchors only; a multi-instance record's anchor came from the
    // union mesh and is not per-instance evidence.
    if (anchor === undefined || m.instanceIds.length !== 1) continue;
    const axis = trustedMeasuredAxis(m.record) ?? declared.axis;
    designBrief.anchorCandidates.push({
      part: m.instanceIds[0]!,
      role: "incremental_declared",
      preferredJointType: declared.jointType ?? "revolute",
      ...(axis !== undefined ? { axis } : {}),
      anchor: roundVec3([anchor[0], anchor[1], anchor[2]]),
      confidence: "high",
      reason: "declared moving by the incremental builder; anchor measured from the as-built assembly mesh",
    });
  }

  designBrief.incrementalPriors = incremental.matched.flatMap((m) => {
    const declared = m.record.declared;
    const parent = m.parentInstanceIds[0] ?? declared.parentPlacedName ?? declared.parentPlanName;
    return m.instanceIds.map((id) => ({
      part: id,
      moving: declared.moving,
      ...(declared.jointType !== undefined ? { jointType: declared.jointType } : {}),
      ...(parent !== undefined ? { parent } : {}),
      ...(declared.axis !== undefined ? { axis: declared.axis } : {}),
      measuredAxis: trustedMeasuredAxis(m.record),
      axisAgrees: m.record.agreement?.axisAgrees ?? null,
    }));
  });
  const incrementalWarnings = [...incremental.sidecarWarnings, ...incremental.warnings, ...axisNotes];
  if (incrementalWarnings.length > 0) designBrief.incrementalWarnings = incrementalWarnings;
}

/** Exported for the incremental-handoff smoke test (pure string builder). */
export function buildPlannerUserPrompt(
  context: {
    objectText: string;
    parts: MotionPartContext[];
    designBrief: MotionDesignBrief;
    renderLegend: string;
    scadContext: string;
  },
  args: {
    partNames: string[];
    incrementalSeed?: MotionPlan | null;
    fixedBase?: boolean;
    defaultCollision?: MotionCollisionApproximation;
  },
): string {
  return [
    "=== OBJECT TEXT ===",
    context.objectText,
    "",
    "=== TASK ===",
    "Study the SCAD code, available top-level modules, assembled part bboxes, assembly placement snippets, and attached render feedback. Design the mechanical articulation for Isaac Sim/OpenUSD Physics.",
    "Return a complete MotionPlan JSON object. Group visual SCAD modules into rigid links, choose joint types, parent/child topology, axes, approximate joint anchors, limits, drives, masses, and collision approximations.",
    "",
    "Important exporter detail: every link Xform is authored at identity and its mesh vertices are already in assembled world coordinates. Therefore localPos0 and localPos1 can usually be the same anchor coordinate in model units.",
    "Use USD/Isaac conventions: +X right, -Y front, +Z up. Revolute/prismatic axis must be X, Y, or Z. Revolute limits are degrees. Prismatic/distance values use model units.",
    args.fixedBase !== undefined ? `Fixed-base override requested: ${args.fixedBase}.` : "Choose fixedBase based on the object; mobile vehicles should usually be floating-base.",
    args.defaultCollision ? `Default collision override requested: ${args.defaultCollision}.` : "Use convexHull unless another supported approximation is clearly better.",
    "",
    "=== DETERMINISTIC MOTION DESIGN BRIEF ===",
    JSON.stringify(context.designBrief, null, 2),
    "",
    "=== AVAILABLE TOP-LEVEL SCAD PARTS ===",
    JSON.stringify(context.parts, null, 2),
    "",
    "=== PER-PART COLOR RENDER LEGEND ===",
    context.renderLegend || "(render legend unavailable; rely on code and part names)",
    "",
    ...(args.incrementalSeed ? [
      "=== DRAFT MOTION PLAN (built from incremental-generation measurements — CORRECT and complete it; do not start from scratch, do not drop declared joints without cause) ===",
      JSON.stringify(args.incrementalSeed, null, 2),
      "",
    ] : []),
    "=== FULL / TRUNCATED SCAD SOURCE ===",
    context.scadContext,
  ].join("\n");
}

function buildAuthorUserPrompt(
  context: {
    objectText: string;
    parts: MotionPartContext[];
    designBrief: MotionDesignBrief;
    renderLegend: string;
    scadContext: string;
  },
  args: {
    partNames: string[];
    fixedBase?: boolean;
    defaultCollision?: MotionCollisionApproximation;
  },
  draftPlan: MotionPlan,
  validation: { ok: boolean; errors: string[]; warnings: string[] },
): string {
  return [
    "=== OBJECT TEXT ===",
    context.objectText,
    "",
    "=== YOUR INPUT DRAFT ARTICULATION PLAN ===",
    JSON.stringify(draftPlan, null, 2),
    "",
    "=== VALIDATOR FEEDBACK FOR THE DRAFT ===",
    JSON.stringify(validation, null, 2),
    "",
    "=== DETERMINISTIC MOTION DESIGN BRIEF ===",
    JSON.stringify(context.designBrief, null, 2),
    "",
    "=== AVAILABLE TOP-LEVEL SCAD PARTS AND GEOMETRY FACTS ===",
    JSON.stringify(context.parts, null, 2),
    "",
    "=== PER-PART COLOR RENDER LEGEND ===",
    context.renderLegend || "(render legend unavailable; rely on code and part names)",
    "",
    "=== TASK ===",
    "Author the final USD Physics motion definition as a strict MotionPlan JSON object. Fix validator errors, preserve all real movable mechanisms, and ensure every joint references valid links and SCAD parts.",
    "This is the definition the deterministic USDA writer will serialize into PhysicsRigidBodyAPI, PhysicsArticulationRootAPI, Physics*Joint prims, PhysicsLimitAPI, PhysicsDriveAPI, and optional PhysxJointAPI.",
    "Do not output prose. Do not output USDA text. Output only the final MotionPlan JSON object.",
    "",
    "Author concrete local joint frames. Because the exporter keeps link Xforms at identity with meshes in assembled coordinates, use localPos0/localPos1 as the shared assembled anchor point unless you have a specific reason to offset child coordinates.",
    args.fixedBase !== undefined ? `Fixed-base override requested: ${args.fixedBase}; obey it.` : "",
    args.defaultCollision ? `Default collision override requested: ${args.defaultCollision}; obey it unless a link explicitly needs another supported collision token.` : "",
    "",
    "=== FULL / TRUNCATED SCAD SOURCE ===",
    context.scadContext,
  ].filter((s) => s !== "").join("\n");
}

function buildRefineUserPrompt(
  context: {
    objectText: string;
    parts: MotionPartContext[];
    designBrief: MotionDesignBrief;
    renderLegend: string;
    scadContext: string;
  },
  args: {
    fixedBase?: boolean;
    defaultCollision?: MotionCollisionApproximation;
  },
  currentPlan: MotionPlan,
  feedback: MotionRefineFeedback,
): string {
  return [
    "=== OBJECT TEXT ===",
    context.objectText,
    "",
    "=== CURRENT MOTION PLAN TO REPAIR ===",
    JSON.stringify(currentPlan, null, 2),
    "",
    "=== DETERMINISTIC FEEDBACK ===",
    JSON.stringify(feedback, null, 2),
    "",
    "=== DETERMINISTIC MOTION DESIGN BRIEF ===",
    JSON.stringify(context.designBrief, null, 2),
    "",
    "=== AVAILABLE TOP-LEVEL SCAD PARTS AND GEOMETRY FACTS ===",
    JSON.stringify(context.parts, null, 2),
    "",
    "=== PER-PART COLOR RENDER LEGEND ===",
    context.renderLegend || "(render legend unavailable; rely on code and part names)",
    "",
    "=== TASK ===",
    "Return the corrected final MotionPlan JSON object. Make the smallest set of changes needed to fix blocking feedback while preserving valid mechanical intent.",
    "Prioritize: validator errors, part coverage, topology, bad anchors, wrong axes, invalid limits/drives, then semantic mechanism quality.",
    "Do not output prose. Do not output USDA. Output only strict JSON.",
    args.fixedBase !== undefined ? `Fixed-base override requested: ${args.fixedBase}; obey it.` : "",
    args.defaultCollision ? `Default collision override requested: ${args.defaultCollision}; obey it unless a link explicitly needs another supported collision token.` : "",
    "",
    "=== FULL / TRUNCATED SCAD SOURCE ===",
    context.scadContext,
  ].filter((s) => s !== "").join("\n");
}

function buildMotionRefineFeedback(
  plan: MotionPlan,
  validation: { ok: boolean; errors: string[]; warnings: string[] },
  context: { objectText: string; parts: MotionPartContext[]; designBrief?: MotionDesignBrief },
  partNames: string[],
): MotionRefineFeedback {
  const linkNames = new Set(plan.links.map((link) => link.name));
  const partNameSet = new Set(partNames);
  const partToLinks = new Map<string, string[]>();
  const unknownParts: Array<{ part: string; link: string }> = [];

  for (const link of plan.links) {
    for (const part of link.parts ?? [link.name]) {
      if (!partNameSet.has(part)) unknownParts.push({ part, link: link.name });
      const links = partToLinks.get(part) ?? [];
      links.push(link.name);
      partToLinks.set(part, links);
    }
  }

  const missingParts = partNames.filter((part) => !partToLinks.has(part));
  const duplicateParts = [...partToLinks.entries()]
    .filter(([part, links]) => partNameSet.has(part) && links.length > 1)
    .map(([part, links]) => ({ part, links }));
  const assignedParts = [...partToLinks.keys()].filter((part) => partNameSet.has(part)).length;

  const linkBboxes = buildPlanLinkBBoxes(plan, context.parts);
  const parentsByChild = new Map<string, string[]>();
  const incomingJointByChild = new Map<string, MotionPlan["joints"][number]>();
  for (const joint of plan.joints) {
    const parents = parentsByChild.get(joint.child) ?? [];
    if (joint.parent) parents.push(joint.parent);
    parentsByChild.set(joint.child, parents);
    if (!incomingJointByChild.has(joint.child)) incomingJointByChild.set(joint.child, joint);
  }

  const rootLink = plan.rootLink ?? plan.links[0]?.name;
  const orphanLinks = plan.links
    .map((link) => link.name)
    .filter((name) => name !== rootLink && (parentsByChild.get(name)?.length ?? 0) === 0);
  const multipleParentLinks = [...parentsByChild.entries()]
    .filter(([, parents]) => new Set(parents).size > 1)
    .map(([link, parents]) => ({ link, parents: [...new Set(parents)] }));
  const cycles = findJointCycles(plan);

  const likelyIssues: string[] = [];
  if (missingParts.length > 0) likelyIssues.push(`Missing SCAD parts in links: ${missingParts.join(", ")}`);
  if (duplicateParts.length > 0) likelyIssues.push(`Duplicate SCAD part assignments: ${duplicateParts.map((d) => d.part).join(", ")}`);
  if (unknownParts.length > 0) likelyIssues.push(`Unknown SCAD parts in plan: ${unknownParts.map((p) => p.part).join(", ")}`);
  if (orphanLinks.length > 0) likelyIssues.push(`Orphan links with no incoming joint: ${orphanLinks.join(", ")}`);
  if (multipleParentLinks.length > 0) likelyIssues.push(`Links with multiple parents: ${multipleParentLinks.map((p) => p.link).join(", ")}`);
  if (cycles.length > 0) likelyIssues.push(`Articulation cycles detected: ${cycles.map((c) => c.join(" -> ")).join("; ")}`);
  if (plan.links.length > 1 && plan.joints.every((joint) => joint.type === "fixed")) {
    likelyIssues.push("All joints are fixed despite multiple links; verify this is not collapsing a movable mechanism");
  }
  if (plan.fixedBase && /\b(rover|vehicle|car|truck|drone|mobile|rolling|wheeled|wheel)\b/i.test(context.objectText)) {
    likelyIssues.push("Object text suggests a mobile/rolling object but fixedBase is true");
  }

  const jointFeedback = plan.joints.map((joint) => {
    const issues: string[] = [];
    const anchor = joint.localPos0 ?? joint.localPos1;
    const parentBox = joint.parent ? linkBboxes.get(joint.parent) : undefined;
    const childBox = linkBboxes.get(joint.child);
    const distanceToParentBBox = anchor && parentBox ? distancePointToBBox(anchor, parentBox) : undefined;
    const distanceToChildBBox = anchor && childBox ? distancePointToBBox(anchor, childBox) : undefined;
    const anchorInsideParent = anchor && parentBox ? pointInsideBBox(anchor, parentBox) : undefined;
    const anchorInsideChild = anchor && childBox ? pointInsideBBox(anchor, childBox) : undefined;

    if (joint.parent && !linkNames.has(joint.parent)) issues.push(`parent link not found: ${joint.parent}`);
    if (!linkNames.has(joint.child)) issues.push(`child link not found: ${joint.child}`);
    if (joint.parent === joint.child) issues.push("parent and child are identical");
    if ((joint.type === "revolute" || joint.type === "prismatic") && !joint.axis) issues.push("axis missing for revolute/prismatic joint");
    if ((joint.type === "revolute" || joint.type === "prismatic") && joint.axis && !["X", "Y", "Z"].includes(joint.axis)) issues.push(`unsupported axis: ${joint.axis}`);
    if (joint.type !== "fixed" && !anchor) issues.push("moving joint has no localPos0/localPos1 anchor");
    if (anchor && parentBox && childBox) {
      const parentDiag = bboxDiagonal(parentBox);
      const childDiag = bboxDiagonal(childBox);
      const suspiciousDistance = Math.max(20, Math.min(parentDiag, childDiag) * 0.75);
      if ((distanceToParentBBox ?? 0) > suspiciousDistance && (distanceToChildBBox ?? 0) > suspiciousDistance) {
        issues.push("anchor is far from both parent and child link bounds");
      }
    }
    if (joint.localPos0 && joint.localPos1 && vecDistance(joint.localPos0, joint.localPos1) > 1) {
      issues.push("localPos0/localPos1 differ even though this exporter usually expects assembled shared anchors");
    }
    const low = limitLowValue(joint.limit);
    const high = limitHighValue(joint.limit);
    const target = joint.drive?.targetPosition;
    if (target !== undefined && low !== undefined && target < low) issues.push("drive targetPosition is below lower limit");
    if (target !== undefined && high !== undefined && target > high) issues.push("drive targetPosition is above upper limit");

    if (issues.length > 0) likelyIssues.push(`Joint ${joint.name}: ${issues.join("; ")}`);
    return {
      name: joint.name,
      type: joint.type,
      ...(joint.parent !== undefined ? { parent: joint.parent } : {}),
      child: joint.child,
      ...(joint.axis !== undefined ? { axis: joint.axis } : {}),
      ...(anchor !== undefined ? { anchor } : {}),
      ...(anchorInsideParent !== undefined ? { anchorInsideParent } : {}),
      ...(anchorInsideChild !== undefined ? { anchorInsideChild } : {}),
      ...(distanceToParentBBox !== undefined ? { distanceToParentBBox: round3(distanceToParentBBox) } : {}),
      ...(distanceToChildBBox !== undefined ? { distanceToChildBBox: round3(distanceToChildBBox) } : {}),
      issues,
    };
  });

  for (const link of plan.links) {
    const text = `${link.name} ${(link.parts ?? []).join(" ")}`;
    const incoming = incomingJointByChild.get(link.name);
    if (/\bwheel|tire|tyre|roller\b/i.test(text) && incoming && incoming.type !== "revolute") {
      likelyIssues.push(`Wheel-like link ${link.name} is attached by ${incoming.type}, expected revolute`);
    }
    if (/\bwing|solar|panel|door|hatch|flap\b/i.test(text) && incoming && incoming.type === "fixed") {
      likelyIssues.push(`Hinge-like link ${link.name} is fixed; verify whether it should be revolute`);
    }
    if (/\bmast|camera|turret|sensor\b/i.test(text) && incoming && incoming.type === "fixed") {
      likelyIssues.push(`Pan/tilt-like link ${link.name} is fixed; verify whether it should move`);
    }
  }

  return {
    validator: validation,
    coverage: {
      totalParts: partNames.length,
      assignedParts,
      missingParts,
      duplicateParts,
      unknownParts,
    },
    topology: {
      ...(rootLink !== undefined ? { rootLink } : {}),
      linkCount: plan.links.length,
      jointCount: plan.joints.length,
      nonFixedJointCount: plan.joints.filter((joint) => joint.type !== "fixed").length,
      orphanLinks,
      multipleParentLinks,
      cycles,
    },
    links: plan.links.map((link) => ({
      name: link.name,
      parts: link.parts ?? [link.name],
      ...(linkBboxes.get(link.name) ? { bbox: linkBboxes.get(link.name)! } : {}),
    })),
    joints: jointFeedback,
    ...(context.designBrief ? {
      semanticExpectations: {
        mechanismHypotheses: context.designBrief.mechanismHypotheses,
        anchorCandidates: context.designBrief.anchorCandidates,
      },
    } : {}),
    likelyIssues: [...new Set(likelyIssues)],
  };
}

/**
 * One motion LLM call: system prompt + text + images in, parsed JSON out.
 *
 * Retries are on the CONTENT, not the transport — `longTimeoutFetch` already
 * owns network retry. What is retried here is a call that returned HTTP 200 and
 * no usable JSON object, which is a transport success and common enough on the
 * Gemini routes to lose a stage.
 */
async function generateMotionJson(args: {
  emitter: StageEmitter;
  route: RouteDef<unknown>;
  modelRef: ModelRef;
  label: string;
  systemPrompt: string;
  userText: string;
  images: MotionImageContext[];
  log: (line: string) => void;
}): Promise<{ raw: string; reasoning: string; parsed: Record<string, unknown> | null }> {
  const userContent: CanonicalPart[] = [{ kind: "text", text: args.userText }];
  for (const img of args.images) {
    userContent.push({ kind: "text", text: img.label });
    userContent.push({ kind: "image", data: img.b64, mimeType: "image/png" });
  }
  args.emitter.emit("message.append", {
    role: "user", stage: args.label, imageCount: args.images.length,
  });

  let raw = "";
  let reasoning = "";
  let parsed: Record<string, unknown> | null = null;
  for (let attempt = 1; attempt <= MOTION_MAX_ATTEMPTS; attempt++) {
    args.log(`  motion: ${args.label} LLM attempt ${attempt}/${MOTION_MAX_ATTEMPTS} via ${args.modelRef.modelId}`);
    let result;
    try {
      const tLLM = Date.now();
      result = await generateOnce({
        route: args.route,
        model: args.modelRef,
        system: args.systemPrompt,
        parts: userContent,
      });
      addStage(`llm.${args.label}`, Date.now() - tLLM);
    } catch (e) {
      args.log(`    motion: ${args.label} attempt ${attempt} threw: ${(e as Error).message}`);
      continue;
    }
    raw = result.text;
    reasoning = result.reasoning;
    const json = extractJsonObject(raw);
    if (!json) {
      args.log(`    motion: ${args.label} attempt ${attempt} returned no JSON object`);
      continue;
    }
    try {
      parsed = JSON.parse(json) as Record<string, unknown>;
      break;
    } catch (e) {
      args.log(`    motion: ${args.label} attempt ${attempt} JSON parse failed: ${(e as Error).message}`);
    }
  }

  args.emitter.emit("message.append", {
    role: "assistant", stage: args.label,
    modelId: args.modelRef.modelId, providerId: args.modelRef.providerId,
  });
  if (raw) args.emitter.emit("part.append", { kind: "text", stage: args.label, text: raw });

  return { raw, reasoning, parsed };
}

function motionPlanFromPayload(payload: Record<string, unknown> | null, partNames: string[]): MotionPlan | null {
  if (!payload) return null;
  const rawPlan = isRecord(payload["plan"]) ? payload["plan"] : payload;
  if (!hasLinksArray(rawPlan)) return null;
  return normalizeMotionPlan(rawPlan, partNames);
}

function loadObjectText(outDir: string): string {
  const paths = [join(outDir, "effective_text.txt"), join(outDir, "prompt.txt")];
  for (const p of paths) {
    if (existsSync(p)) return readFileSync(p, "utf8").trim();
  }
  return "(no original text prompt available)";
}

function loadPartDescriptions(outDir: string): Map<string, string> {
  const out = new Map<string, string>();
  const p = join(outDir, "plan.json");
  if (!existsSync(p)) return out;
  try {
    const raw = JSON.parse(readFileSync(p, "utf8")) as unknown;
    if (!Array.isArray(raw)) return out;
    for (const item of raw) {
      if (!isRecord(item)) continue;
      const name = typeof item["name"] === "string" ? item["name"] : "";
      const desc = typeof item["description"] === "string" ? item["description"] : "";
      if (name && desc) out.set(name, desc);
    }
  } catch {
    return out;
  }
  return out;
}

function scadContext(scadCode: string): string {
  if (scadCode.length <= MAX_SCAD_CONTEXT_CHARS) return scadCode;
  const head = Math.floor(MAX_SCAD_CONTEXT_CHARS * 0.7);
  const tail = MAX_SCAD_CONTEXT_CHARS - head;
  return (
    scadCode.slice(0, head) +
    `\n\n// ... SCAD truncated for motion-planning context (${scadCode.length} chars total) ...\n\n` +
    scadCode.slice(scadCode.length - tail)
  );
}

function compactWhitespace(text: string, maxChars: number): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > maxChars ? `${compact.slice(0, maxChars)}...` : compact;
}

function enrichMotionParts(parts: MotionPartContext[]): MotionPartContext[] {
  return parts.map((part) => {
    const semanticTags = inferSemanticTags(part);
    const motionRole = inferMotionRole(part, semanticTags);
    const motionHint = buildPartMotionHint(part, motionRole, semanticTags);
    const bbox = part.bbox;
    return {
      ...part,
      ...(bbox ? { center: bboxCenter(bbox) } : {}),
      ...(bbox ? { diagonal: round3(bboxDiagonal(bbox)) } : {}),
      ...(bbox ? { volume: round3(Math.max(0, bbox.size[0] * bbox.size[1] * bbox.size[2])) } : {}),
      semanticTags,
      motionRole,
      motionHint,
    };
  });
}

function buildMotionDesignBrief(args: {
  objectText: string;
  parts: MotionPartContext[];
}): MotionDesignBrief {
  const bboxParts = args.parts.filter((part) => part.bbox);
  const modelBox = bboxParts.length > 0
    ? mergeBBoxes(bboxParts.map((part) => part.bbox!))
    : undefined;
  const modelCenter = modelBox ? bboxCenter(modelBox) : [0, 0, 0] as [number, number, number];
  const likelyMobileBase = /\b(rover|vehicle|car|truck|drone|mobile|rolling|wheeled|wheel|walker)\b/i.test(args.objectText);
  const likelyStaticFixture = /\b(fixtur|machine|arm|mount|stand|tripod|crane|gate|door|robot arm|industrial)\b/i.test(args.objectText);

  return {
    summary: {
      partCount: args.parts.length,
      bboxAvailable: bboxParts.length,
      likelyMobileBase,
      likelyStaticFixture,
    },
    coordinateSystem: {
      x: "+X is right",
      y: "-Y is front, +Y is back",
      z: "+Z is up",
      units: "model units; metersPerUnit is usually 0.001",
      jointFrameNote: "Link Xforms are identity and link meshes are in assembled coordinates, so localPos0/localPos1 usually use the same assembled anchor.",
    },
    partRoles: args.parts.map((part) => ({
      part: part.name,
      role: part.motionRole ?? "unknown",
      semanticTags: part.semanticTags ?? [],
      ...(part.center ? { center: roundVec3(part.center) } : {}),
      ...(part.bbox ? { size: roundVec3(part.bbox.size) } : {}),
      motionHint: part.motionHint,
    })),
    symmetryGroups: buildSymmetryGroups(args.parts),
    proximityPairs: buildProximityPairs(args.parts, modelBox),
    anchorCandidates: buildAnchorCandidates(args.parts, modelCenter),
    mechanismHypotheses: buildMechanismHypotheses(args.parts, likelyMobileBase),
    plannerChecklist: [
      "Use the mechanism hypotheses as evidence, not as a forced plan.",
      "Group visual/cosmetic modules into rigid links when they should move together.",
      "Use symmetry groups to keep left/right or front/rear joint axes, limits, masses, and drives consistent.",
      "Prefer anchor candidates near physical pivots, axles, hinges, telescoping axes, and panel roots.",
      "If an anchor candidate conflicts with render evidence or SCAD placement, choose the physical pivot visible in the render.",
      "Every SCAD top-level module should appear in exactly one link unless intentionally omitted for a clear reason.",
      "Mobile bases should usually use fixedBase=false; anchored fixtures should usually use fixedBase=true.",
      "For closed loops, keep the articulation tree and mark loop-closing constraints excluded from articulation.",
    ],
  };
}

function inferSemanticTags(part: MotionPartContext): string[] {
  const text = `${part.name} ${part.description ?? ""}`.toLowerCase();
  const tags: string[] = [];
  const checks: Array<[string, RegExp]> = [
    ["root_body", /\b(chassis|body|base|frame|deck|hull|housing|platform)\b/],
    ["wheel", /\b(wheel|tire|tyre|roller|axle)\b/],
    ["track", /\b(track|tread|crawler)\b/],
    ["rocker", /\b(rocker|suspension|swingarm)\b/],
    ["bogie", /\b(bogie|truck|trailing arm)\b/],
    ["panel", /\b(panel|solar|wing|array|flap|fold)\b/],
    ["mast", /\b(mast|neck|pan|turret|pedestal)\b/],
    ["camera", /\b(camera|sensor|head|lens|lidar|radar)\b/],
    ["arm", /\b(arm|boom|linkage|elbow|shoulder|wrist)\b/],
    ["slider", /\b(slider|slide|rail|drawer|piston|telescop|linear|rod)\b/],
    ["door", /\b(door|hatch|gate|lid|cover)\b/],
    ["gear", /\b(gear|pinion|rack|sprocket)\b/],
    ["decor", /\b(vent|chimney|trim|bolt|rivet|handle|knob|ornament|grille|grill)\b/],
    ["left", /\bleft\b|^left_|_left\b/],
    ["right", /\bright\b|^right_|_right\b/],
    ["front", /\bfront\b|^front_|_front\b/],
    ["rear", /\b(rear|back)\b|^rear_|_rear\b|^back_|_back\b/],
  ];
  for (const [tag, pattern] of checks) {
    if (pattern.test(text)) tags.push(tag);
  }
  return [...new Set(tags)];
}

function inferMotionRole(part: MotionPartContext, tags: string[]): string {
  if (tags.includes("root_body")) return "root_body";
  if (tags.includes("wheel")) return "rolling_wheel";
  if (tags.includes("track")) return "track_or_tread";
  if (tags.includes("rocker")) return "rocker_suspension";
  if (tags.includes("bogie")) return "bogie_suspension";
  if (tags.includes("panel")) return "hinged_panel";
  if (tags.includes("mast")) return "pan_or_mast";
  if (tags.includes("camera")) return "tilt_sensor_head";
  if (tags.includes("slider")) return "linear_slider";
  if (tags.includes("door")) return "hinged_door_or_hatch";
  if (tags.includes("gear")) return "coupled_gear";
  if (tags.includes("arm")) return "articulated_arm";
  if (tags.includes("decor")) return "fixed_detail";
  if (part.bbox) {
    const [x, y, z] = part.bbox.size;
    const max = Math.max(x, y, z);
    const min = Math.max(1e-6, Math.min(x, y, z));
    if (max / min > 8) return "long_rod_or_beam";
    if (z < Math.max(x, y) * 0.12) return "thin_plate";
  }
  return "unknown";
}

function buildPartMotionHint(
  part: MotionPartContext,
  role: string,
  tags: string[],
): NonNullable<MotionPartContext["motionHint"]> {
  const preferredJointTypes: MotionJointSpec["type"][] = [];
  const likelyAxes: MotionAxis[] = [];
  let roleReason = "No strong semantic motion cue; inspect render and SCAD placement.";

  if (role === "root_body" || role === "fixed_detail") {
    preferredJointTypes.push("fixed");
    roleReason = "Body/decorative parts are usually rigidly grouped with the nearest structural link.";
  } else if (role === "rolling_wheel" || role === "track_or_tread") {
    preferredJointTypes.push("revolute");
    likelyAxes.push("X");
    roleReason = "Wheel/roller/tread terms indicate an axle rotation joint.";
  } else if (role === "rocker_suspension" || role === "bogie_suspension") {
    preferredJointTypes.push("revolute");
    likelyAxes.push("X");
    roleReason = "Rocker/bogie suspension parts typically pivot around a lateral axle.";
  } else if (role === "hinged_panel") {
    preferredJointTypes.push("revolute");
    likelyAxes.push(tags.includes("left") || tags.includes("right") ? "Y" : "X");
    roleReason = "Panel/wing/flap terms usually imply a hinge at the root edge.";
  } else if (role === "pan_or_mast") {
    preferredJointTypes.push("revolute");
    likelyAxes.push("Z");
    roleReason = "Mast/turret/pan terms usually rotate around a vertical axis.";
  } else if (role === "tilt_sensor_head") {
    preferredJointTypes.push("revolute");
    likelyAxes.push("X", "Y");
    roleReason = "Camera/sensor heads commonly tilt on a horizontal axis.";
  } else if (role === "linear_slider") {
    preferredJointTypes.push("prismatic");
    likelyAxes.push(dominantAxis(part.bbox));
    roleReason = "Slider/rail/piston terms imply translation along the dominant linear axis.";
  } else if (role === "hinged_door_or_hatch") {
    preferredJointTypes.push("revolute");
    likelyAxes.push("Z", "X");
    roleReason = "Door/hatch/lid terms imply a hinge along one edge.";
  } else if (role === "coupled_gear") {
    preferredJointTypes.push("gear", "revolute");
    likelyAxes.push("Z", "X");
    roleReason = "Gear/rack/pinion terms may need coupled-joint metadata plus an underlying revolute/prismatic joint.";
  } else if (role === "articulated_arm" || role === "long_rod_or_beam") {
    preferredJointTypes.push("revolute", "prismatic");
    likelyAxes.push(dominantAxis(part.bbox));
    roleReason = "Arm/rod/beam parts often connect through revolute pivots or linear telescoping joints.";
  }

  return {
    preferredJointTypes: preferredJointTypes.length > 0 ? preferredJointTypes : ["fixed", "revolute"],
    likelyAxes: likelyAxes.length > 0 ? [...new Set(likelyAxes)] : ["X", "Y", "Z"],
    roleReason,
  };
}

function buildSymmetryGroups(parts: MotionPartContext[]): MotionDesignBrief["symmetryGroups"] {
  const groups: MotionDesignBrief["symmetryGroups"] = [];
  for (const kind of ["left_right", "front_rear"] as const) {
    const byKey = new Map<string, Partial<Record<"negative" | "positive", MotionPartContext>>>();
    for (const part of parts) {
      const side = symmetryKey(part.name, kind);
      if (!side) continue;
      const entry = byKey.get(side.key) ?? {};
      entry[side.side] = part;
      byKey.set(side.key, entry);
    }
    for (const entry of byKey.values()) {
      if (!entry.negative || !entry.positive) continue;
      const neg = entry.negative;
      const pos = entry.positive;
      const negCenter = neg.center ?? [0, 0, 0];
      const posCenter = pos.center ?? [0, 0, 0];
      groups.push({
        kind,
        negative: neg.name,
        positive: pos.name,
        mirroredAxis: kind === "left_right" ? "X" : "Y",
        centerDelta: roundVec3([
          posCenter[0] - negCenter[0],
          posCenter[1] - negCenter[1],
          posCenter[2] - negCenter[2],
        ]),
      });
    }
  }
  return groups;
}

function symmetryKey(name: string, kind: "left_right" | "front_rear"): { key: string; side: "negative" | "positive" } | null {
  const lower = name.toLowerCase();
  if (kind === "left_right") {
    if (lower.startsWith("left_")) return { key: lower.slice(5), side: "negative" };
    if (lower.startsWith("right_")) return { key: lower.slice(6), side: "positive" };
    if (lower.endsWith("_left")) return { key: lower.slice(0, -5), side: "negative" };
    if (lower.endsWith("_right")) return { key: lower.slice(0, -6), side: "positive" };
  } else {
    if (lower.startsWith("front_")) return { key: lower.slice(6), side: "negative" };
    if (lower.startsWith("rear_")) return { key: lower.slice(5), side: "positive" };
    if (lower.startsWith("back_")) return { key: lower.slice(5), side: "positive" };
    if (lower.endsWith("_front")) return { key: lower.slice(0, -6), side: "negative" };
    if (lower.endsWith("_rear")) return { key: lower.slice(0, -5), side: "positive" };
    if (lower.endsWith("_back")) return { key: lower.slice(0, -5), side: "positive" };
  }
  return null;
}

function buildProximityPairs(
  parts: MotionPartContext[],
  modelBox: NonNullable<MotionPartContext["bbox"]> | undefined,
): MotionDesignBrief["proximityPairs"] {
  const withBoxes = parts.filter((part) => part.bbox && part.center);
  const modelDiag = modelBox ? bboxDiagonal(modelBox) : 100;
  const maxGap = Math.max(8, modelDiag * 0.08);
  const pairs: MotionDesignBrief["proximityPairs"] = [];
  for (let i = 0; i < withBoxes.length; i++) {
    for (let j = i + 1; j < withBoxes.length; j++) {
      const a = withBoxes[i]!;
      const b = withBoxes[j]!;
      const gap = bboxGap(a.bbox!, b.bbox!);
      if (gap > maxGap) continue;
      const centerDistance = vecDistance(a.center!, b.center!);
      pairs.push({
        a: a.name,
        b: b.name,
        bboxGap: round3(gap),
        centerDistance: round3(centerDistance),
        relation: gap <= 1 ? "touching_or_intersecting" : "nearby",
      });
    }
  }
  return pairs
    .sort((a, b) => a.bboxGap - b.bboxGap || a.centerDistance - b.centerDistance)
    .slice(0, 64);
}

function buildAnchorCandidates(
  parts: MotionPartContext[],
  modelCenter: [number, number, number],
): MotionDesignBrief["anchorCandidates"] {
  const out: MotionDesignBrief["anchorCandidates"] = [];
  for (const part of parts) {
    if (!part.bbox || !part.center || !part.motionHint) continue;
    const role = part.motionRole ?? "unknown";
    const preferredJointType = part.motionHint.preferredJointTypes.find((type) => type !== "fixed") ?? part.motionHint.preferredJointTypes[0] ?? "fixed";
    if (preferredJointType === "fixed") continue;
    const axis = part.motionHint.likelyAxes[0];
    const anchor = anchorForRole(part, modelCenter);
    out.push({
      part: part.name,
      role,
      preferredJointType,
      ...(axis ? { axis } : {}),
      anchor: roundVec3(anchor),
      confidence: anchorConfidence(role),
      reason: anchorReason(role),
    });
  }
  return out.slice(0, 96);
}

function buildMechanismHypotheses(parts: MotionPartContext[], likelyMobileBase: boolean): MotionDesignBrief["mechanismHypotheses"] {
  const byRole = new Map<string, string[]>();
  for (const part of parts) {
    const role = part.motionRole ?? "unknown";
    const list = byRole.get(role) ?? [];
    list.push(part.name);
    byRole.set(role, list);
  }
  const out: MotionDesignBrief["mechanismHypotheses"] = [];
  const wheels = byRole.get("rolling_wheel") ?? [];
  if (wheels.length > 0) {
    out.push({
      name: likelyMobileBase ? "mobile rolling base" : "wheel or roller assembly",
      evidenceParts: wheels,
      recommendedTopology: "Root chassis/body link with each wheel as a revolute child; keep symmetric wheel drives consistent.",
      recommendedJointTypes: ["revolute"],
      notes: ["Wheel anchors should sit at the wheel centers/axles.", "Use floating base for rovers/vehicles unless explicitly anchored."],
    });
  }
  const suspension = [...(byRole.get("rocker_suspension") ?? []), ...(byRole.get("bogie_suspension") ?? [])];
  if (suspension.length > 0) {
    out.push({
      name: "rocker-bogie or pivoting suspension",
      evidenceParts: suspension,
      recommendedTopology: "Attach rocker links to chassis with revolute pivots; attach bogies to rockers with revolute pivots; wheels attach below suspension links.",
      recommendedJointTypes: ["revolute"],
      notes: ["Use matched limits left/right.", "Avoid cycles unless loop-closing constraints are excluded from articulation."],
    });
  }
  const panels = byRole.get("hinged_panel") ?? [];
  if (panels.length > 0) {
    out.push({
      name: "deployable panels or wings",
      evidenceParts: panels,
      recommendedTopology: "Panel links should hinge from the body/root at their inboard edge.",
      recommendedJointTypes: ["revolute"],
      notes: ["Use +/- 90 degree limits for deployable panels unless the object implies a smaller travel."],
    });
  }
  const mastParts = [...(byRole.get("pan_or_mast") ?? []), ...(byRole.get("tilt_sensor_head") ?? [])];
  if (mastParts.length > 0) {
    out.push({
      name: "sensor mast pan/tilt",
      evidenceParts: mastParts,
      recommendedTopology: "Base mast pan joint around Z, followed by camera/head tilt joint around a horizontal axis.",
      recommendedJointTypes: ["revolute"],
      notes: ["Separate pan and tilt links when the geometry has distinct mast and head modules."],
    });
  }
  const sliders = byRole.get("linear_slider") ?? [];
  if (sliders.length > 0) {
    out.push({
      name: "linear slider or telescoping mechanism",
      evidenceParts: sliders,
      recommendedTopology: "Use prismatic joints along the rail/piston dominant axis.",
      recommendedJointTypes: ["prismatic"],
      notes: ["Set lower/upper travel in model units and keep drive target within limits."],
    });
  }
  const coupled = byRole.get("coupled_gear") ?? [];
  if (coupled.length > 0) {
    out.push({
      name: "coupled gear/rack mechanism",
      evidenceParts: coupled,
      recommendedTopology: "Represent the physical articulation with revolute/prismatic joints and add gear/rack/mimic metadata only where coupling is important.",
      recommendedJointTypes: ["revolute", "gear", "rack_and_pinion", "mimic"],
      notes: ["Coupled joints should include referenceJoint and ratio."],
    });
  }
  return out;
}

function anchorForRole(part: MotionPartContext, modelCenter: [number, number, number]): [number, number, number] {
  const bbox = part.bbox!;
  const center = part.center ?? bboxCenter(bbox);
  const role = part.motionRole ?? "unknown";
  if (role === "rolling_wheel" || role === "rocker_suspension" || role === "bogie_suspension") {
    return center;
  }
  if (role === "hinged_panel") {
    const x = center[0] < modelCenter[0] ? bbox.max[0] : bbox.min[0];
    return [x, center[1], center[2]];
  }
  if (role === "pan_or_mast") {
    return [center[0], center[1], bbox.min[2]];
  }
  if (role === "tilt_sensor_head") {
    return [center[0], bbox.min[1], center[2]];
  }
  if (role === "linear_slider") {
    return center;
  }
  return nearestFaceAnchor(bbox, modelCenter);
}

function nearestFaceAnchor(
  bbox: NonNullable<MotionPartContext["bbox"]>,
  target: [number, number, number],
): [number, number, number] {
  const c = bboxCenter(bbox);
  const candidates: [number, number, number][] = [
    [bbox.min[0], c[1], c[2]],
    [bbox.max[0], c[1], c[2]],
    [c[0], bbox.min[1], c[2]],
    [c[0], bbox.max[1], c[2]],
    [c[0], c[1], bbox.min[2]],
    [c[0], c[1], bbox.max[2]],
  ];
  return candidates.sort((a, b) => vecDistance(a, target) - vecDistance(b, target))[0]!;
}

function anchorConfidence(role: string): "high" | "medium" | "low" {
  if (role === "rolling_wheel" || role === "pan_or_mast" || role === "linear_slider") return "high";
  if (role === "hinged_panel" || role === "rocker_suspension" || role === "bogie_suspension" || role === "tilt_sensor_head") return "medium";
  return "low";
}

function anchorReason(role: string): string {
  if (role === "rolling_wheel") return "Wheel-like part: axle is usually at the part center.";
  if (role === "hinged_panel") return "Panel-like part: hinge is likely on the inboard edge closest to the body.";
  if (role === "pan_or_mast") return "Mast/turret-like part: pan pivot is usually at the lower center.";
  if (role === "tilt_sensor_head") return "Sensor-head-like part: tilt pivot is near the head support side.";
  if (role === "linear_slider") return "Slider-like part: prismatic axis is usually centered along the rail/piston.";
  return "Generic candidate: nearest face center toward the model body.";
}

function dominantAxis(bbox: MotionPartContext["bbox"] | undefined): MotionAxis {
  if (!bbox) return "X";
  const sizes = bbox.size;
  if (sizes[1] > sizes[0] && sizes[1] >= sizes[2]) return "Y";
  if (sizes[2] > sizes[0] && sizes[2] > sizes[1]) return "Z";
  return "X";
}

function bboxCenter(bbox: NonNullable<MotionPartContext["bbox"]>): [number, number, number] {
  return [
    (bbox.min[0] + bbox.max[0]) * 0.5,
    (bbox.min[1] + bbox.max[1]) * 0.5,
    (bbox.min[2] + bbox.max[2]) * 0.5,
  ];
}

function bboxGap(a: NonNullable<MotionPartContext["bbox"]>, b: NonNullable<MotionPartContext["bbox"]>): number {
  const dx = Math.max(a.min[0] - b.max[0], b.min[0] - a.max[0], 0);
  const dy = Math.max(a.min[1] - b.max[1], b.min[1] - a.max[1], 0);
  const dz = Math.max(a.min[2] - b.max[2], b.min[2] - a.max[2], 0);
  return Math.hypot(dx, dy, dz);
}

function roundVec3(value: [number, number, number]): [number, number, number] {
  return [round3(value[0]), round3(value[1]), round3(value[2])];
}

function buildPlanLinkBBoxes(
  plan: MotionPlan,
  parts: MotionPartContext[],
): Map<string, NonNullable<MotionPartContext["bbox"]>> {
  const byPart = new Map(parts.filter((part) => part.bbox).map((part) => [part.name, part.bbox!] as const));
  const out = new Map<string, NonNullable<MotionPartContext["bbox"]>>();
  for (const link of plan.links) {
    const boxes = (link.parts ?? [link.name]).map((part) => byPart.get(part)).filter((box): box is NonNullable<MotionPartContext["bbox"]> => Boolean(box));
    if (boxes.length === 0) continue;
    out.set(link.name, mergeBBoxes(boxes));
  }
  return out;
}

function mergeBBoxes(boxes: Array<NonNullable<MotionPartContext["bbox"]>>): NonNullable<MotionPartContext["bbox"]> {
  const min: [number, number, number] = [
    Math.min(...boxes.map((box) => box.min[0])),
    Math.min(...boxes.map((box) => box.min[1])),
    Math.min(...boxes.map((box) => box.min[2])),
  ];
  const max: [number, number, number] = [
    Math.max(...boxes.map((box) => box.max[0])),
    Math.max(...boxes.map((box) => box.max[1])),
    Math.max(...boxes.map((box) => box.max[2])),
  ];
  return {
    min,
    max,
    size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
  };
}

function pointInsideBBox(point: [number, number, number], bbox: NonNullable<MotionPartContext["bbox"]>): boolean {
  const eps = 1e-4;
  return point[0] >= bbox.min[0] - eps && point[0] <= bbox.max[0] + eps &&
    point[1] >= bbox.min[1] - eps && point[1] <= bbox.max[1] + eps &&
    point[2] >= bbox.min[2] - eps && point[2] <= bbox.max[2] + eps;
}

function distancePointToBBox(point: [number, number, number], bbox: NonNullable<MotionPartContext["bbox"]>): number {
  const dx = Math.max(bbox.min[0] - point[0], 0, point[0] - bbox.max[0]);
  const dy = Math.max(bbox.min[1] - point[1], 0, point[1] - bbox.max[1]);
  const dz = Math.max(bbox.min[2] - point[2], 0, point[2] - bbox.max[2]);
  return Math.hypot(dx, dy, dz);
}

function bboxDiagonal(bbox: NonNullable<MotionPartContext["bbox"]>): number {
  return Math.hypot(bbox.size[0], bbox.size[1], bbox.size[2]);
}

function vecDistance(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function limitLowValue(limit: MotionPlan["joints"][number]["limit"]): number | undefined {
  return limit?.lower ?? limit?.low;
}

function limitHighValue(limit: MotionPlan["joints"][number]["limit"]): number | undefined {
  return limit?.upper ?? limit?.high;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function findJointCycles(plan: MotionPlan): string[][] {
  const byParent = new Map<string, string[]>();
  for (const joint of plan.joints) {
    if (!joint.parent) continue;
    const children = byParent.get(joint.parent) ?? [];
    children.push(joint.child);
    byParent.set(joint.parent, children);
  }

  const cycles: string[][] = [];
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const stack: string[] = [];

  function visit(node: string): void {
    if (visiting.has(node)) {
      const start = stack.indexOf(node);
      if (start !== -1) cycles.push([...stack.slice(start), node]);
      return;
    }
    if (visited.has(node)) return;
    visiting.add(node);
    stack.push(node);
    for (const child of byParent.get(node) ?? []) visit(child);
    stack.pop();
    visiting.delete(node);
    visited.add(node);
  }

  for (const link of plan.links) visit(link.name);
  return dedupeCycles(cycles);
}

function dedupeCycles(cycles: string[][]): string[][] {
  const seen = new Set<string>();
  const out: string[][] = [];
  for (const cycle of cycles) {
    const key = cycle.join(">");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(cycle);
  }
  return out;
}

function extractJsonObject(text: string): string | null {
  const fenced = /```(?:json)?\s*\n?([\s\S]*?)\n?```/g;
  for (let m: RegExpExecArray | null; (m = fenced.exec(text)); ) {
    const body = m[1]!.trim();
    if (body.startsWith("{")) return body;
  }
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nextId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function hasLinksArray(value: unknown): boolean {
  return typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    Array.isArray((value as { links?: unknown }).links);
}

function safePathName(value: string): string {
  const s = value.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "");
  return s || "link";
}
