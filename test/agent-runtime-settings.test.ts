import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { JsStore } from '@animalabs/chronicle';

import { AutobiographicalStrategy, ContextManager, type ContextInjection, type StrategyContext } from '@animalabs/context-manager';
import type { Membrane, NormalizedRequest } from '@animalabs/membrane';
import { Agent, AgentFramework, BudgetPreflightError } from '../src/index.js';

const membrane = {} as any;

function strategy(): AutobiographicalStrategy {
  return new AutobiographicalStrategy({
    adaptiveResolution: true,
    foldingStrategy: 'kv-stable',
    recentWindowTokens: 30_000,
    kvStableReachTokens: 8_000,
  });
}

describe('agent runtime settings', () => {
it('native memory ground initializes before activation, follows composed real activation, and ignores pure preview', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-memory-ground-'));
  const ground = 'Disposable full carrier ground.';
  const frame = 'Disposable exact parent framing.';
  const requests: NormalizedRequest[] = [];
  const nativeMembrane = {
    complete: async (request: NormalizedRequest) => {
      requests.push(structuredClone(request));
      return { content: [{ type: 'text', text: 'Disposable native structural output, not real recollection.' }], stopReason: 'end_turn', usage: { inputTokens: 100, outputTokens: 20 } };
    },
    streamYielding: () => ({}) as never,
  } as unknown as Membrane;
  const nativeToolStates: unknown[] = [];
  class MemoryGroundStrategy extends AutobiographicalStrategy {
    override async tick(ctx: StrategyContext): Promise<void> { nativeToolStates.push(ctx.tools); await super.tick(ctx); }
  }
  const cm = await ContextManager.open({ path: join(dir, 'store'), namespace: 'agents/liv', membrane: nativeMembrane, strategy: new MemoryGroundStrategy({
    compressionModel: 'gpt-6.1-sol', targetChunkTokens: 24_000, headWindowTokens: 512, recentWindowTokens: 150_000,
    summaryTargetTokens: 1536, compressionMaxTokens: 8192, mergeThreshold: 4, l1HoldbackChunks: 0, adaptiveResolution: true,
    foldingStrategy: 'kv-stable', identityReminder: frame, autoTickOnNewMessage: false, logEffectiveConfig: false,
  }) });
  const agent = new Agent({ name: 'liv', model: 'gpt-6.1-sol', systemPrompt: ground, contextBudgetTokens: 604608 }, cm, nativeMembrane);
  const injection = (value: string): ContextInjection[] => [{ namespace: 'disposable-ground-proof', position: 'system', content: [{ type: 'text', text: value }] }];
  const tools = [{ name: 'fixture_tool', description: 'Inert, never executed.', inputSchema: { type: 'object' as const } }];
  const installedTools: unknown[] = [];
  const installTools = cm.setToolDefinitions.bind(cm);
  cm.setToolDefinitions = value => { installedTools.push(value); installTools(value); };
  try {
    const preview = await agent.buildActivationRequest(tools, injection('PREVIEW-ONLY SYSTEM DATA'));
    assert.equal(preview.system, `${ground}\nPREVIEW-ONLY SYSTEM DATA`); assert.equal(requests.length, 0);
    const first = cm.addMessage('user', [{ type: 'text', text: 'Disposable preactivation historical observation.' }]);
    cm.finalizeArchivalBatch(first); await cm.tick();
    assert.equal(requests[0].system, ground, 'constructor initializes native ground; preview cannot replace it');
    assert.deepEqual(installedTools, [], 'preview never calls the live tool setter');
    assert.equal(nativeToolStates[0], undefined, 'preview never installs live CM tools');
    const activation = await agent.startStreamWithInjections(tools, injection('REAL ACTIVATION SYSTEM DATA'));
    assert.equal(activation.request.system, `${ground}\nREAL ACTIVATION SYSTEM DATA`);
    assert.deepEqual(installedTools, [tools], 'actual activation installs its live tools exactly once');
    assert.ok(!JSON.stringify(activation.request).includes(frame), 'deliberate memory framing never governs the ordinary primary request');
    const second = cm.addMessage('liv', [{ type: 'text', text: 'Disposable postactivation historical observation.' }]);
    cm.finalizeArchivalBatch(second); await cm.tick();
    assert.equal(requests[1].system, activation.request.system, 'real activation refreshes the composed native ground');
    assert.deepEqual(nativeToolStates[1], tools, 'actual activation updates the live strategy context');
    await agent.buildActivationRequest([{ ...tools[0], name: 'preview_only_tool' }], injection('LATER PREVIEW ONLY'));
    assert.deepEqual(installedTools, [tools], 'later preview cannot replace an actual tool surface');
    await cm.resetHeadWindow();
    assert.equal(requests.length, 3); assert.equal(requests[2].system, activation.request.system, 'auxiliary memory retains actual activation, not preview');
    for (const request of requests) {
      assert.equal(request.config.model, 'gpt-6.1-sol'); assert.equal(request.tools, undefined);
      assert.ok(request.messages.some(message => message.content.some(block => block.type === 'text' && block.text.includes(frame))));
    }
  } finally { agent.reset(); cm.close(); rmSync(dir, { recursive: true, force: true }); }
});
it('agent_settings is one typed tool for the hot runtime surface', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-'));
  const framework = await AgentFramework.create({
    storePath: join(dir, 'store'),
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      strategy: strategy(),
      contextBudgetTokens: 100_000,
      maxTokens: 10_000,
    }],
    modules: [],
  });
  try {
    const tool = framework.getAllTools().find((candidate) => candidate.name === 'agent_settings');
    assert.ok(tool, 'general-purpose settings tool is exposed');
    assert.deepEqual(
      (tool!.inputSchema as { properties: Record<string, unknown> }).properties.same_round_think_text_policy,
      {
        type: 'string',
        enum: ['public', 'private'],
        description:
          'Routing policy for ordinary text emitted in the same native assistant round as think(). ' +
          "Omitted in the recipe preserves the compatibility carry-forward: public.",
      },
    );
    assert.deepEqual(framework.getAgentRuntimeSettings('agent'), {
      contextBudgetTokens: 100_000,
      tailTokens: 30_000,
      transitionPaceTokens: 8_000,
      sameRoundThinkTextPolicy: 'public',
      sameRoundThinkTextPolicySource: 'compatibility_default',
      transition: 'stable',
    });

    assert.deepEqual(
      framework.updateAgentRuntimeSettings('agent', {
        contextBudgetTokens: 60_000,
        tailTokens: 20_000,
        transitionPaceTokens: 4_000,
        sameRoundThinkTextPolicy: 'private',
      }),
      {
        contextBudgetTokens: 60_000,
        tailTokens: 20_000,
        transitionPaceTokens: 4_000,
        sameRoundThinkTextPolicy: 'private',
        sameRoundThinkTextPolicySource: 'runtime_override',
        transition: 'converging',
      },
    );
    assert.equal(
      framework.cancelAgentRuntimeSettingsTransition('agent').transition,
      'stable',
    );
    assert.deepEqual(framework.resetAgentRuntimeSettings('agent'), {
      contextBudgetTokens: 100_000,
      tailTokens: 30_000,
      transitionPaceTokens: 8_000,
      sameRoundThinkTextPolicy: 'public',
      sameRoundThinkTextPolicySource: 'compatibility_default',
      transition: 'stable',
    });
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

it('runtime overrides persist across framework restart', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-persist-'));
  const storePath = join(dir, 'store');
  const config = () => ({
    storePath,
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      strategy: strategy(),
      contextBudgetTokens: 100_000,
      maxTokens: 10_000,
      sameRoundThinkTextPolicy: 'public' as const,
    }],
    modules: [],
  });

  let framework = await AgentFramework.create(config());
  framework.updateAgentRuntimeSettings('agent', {
    contextBudgetTokens: 70_000,
    tailTokens: 18_000,
    transitionPaceTokens: 5_000,
    sameRoundThinkTextPolicy: 'private',
  });
  await framework.stop();

  try {
    framework = await AgentFramework.create(config());
    assert.deepEqual(framework.getAgentRuntimeSettings('agent'), {
      contextBudgetTokens: 70_000,
      tailTokens: 18_000,
      transitionPaceTokens: 5_000,
      sameRoundThinkTextPolicy: 'private',
      sameRoundThinkTextPolicySource: 'runtime_override',
      transition: 'converging',
    });
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

it('reset-all skips tail controls unsupported by the active strategy', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-basic-'));
  const framework = await AgentFramework.create({
    storePath: join(dir, 'store'),
    membrane,
    agents: [{ name: 'agent', model: 'test-model', systemPrompt: 'test' }],
    modules: [],
  });
  try {
    assert.deepEqual(framework.resetAgentRuntimeSettings('agent'), {
      contextBudgetTokens: 100_000,
      sameRoundThinkTextPolicy: 'public',
      sameRoundThinkTextPolicySource: 'compatibility_default',
      transition: 'stable',
    });
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

it('same_round_think_text_policy reports recipe/runtime/default sources and rejects invalid values', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-think-policy-'));
  const framework = await AgentFramework.create({
    storePath: join(dir, 'store'),
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      sameRoundThinkTextPolicy: 'private',
    }],
    modules: [],
  });
  try {
    assert.deepEqual(framework.getAgentRuntimeSettings('agent'), {
      contextBudgetTokens: 100_000,
      sameRoundThinkTextPolicy: 'private',
      sameRoundThinkTextPolicySource: 'recipe',
      transition: 'stable',
    });

    assert.deepEqual(
      framework.updateAgentRuntimeSettings('agent', { sameRoundThinkTextPolicy: 'public' }),
      {
        contextBudgetTokens: 100_000,
        sameRoundThinkTextPolicy: 'public',
        sameRoundThinkTextPolicySource: 'runtime_override',
        transition: 'stable',
      },
    );

    assert.deepEqual(
      framework.resetAgentRuntimeSettings('agent', ['sameRoundThinkTextPolicy']),
      {
        contextBudgetTokens: 100_000,
        sameRoundThinkTextPolicy: 'private',
        sameRoundThinkTextPolicySource: 'recipe',
        transition: 'stable',
      },
    );

    assert.throws(
      () => framework.updateAgentRuntimeSettings('agent', { sameRoundThinkTextPolicy: 'bogus' as 'public' }),
      /sameRoundThinkTextPolicy must be 'public' or 'private'/,
    );
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

it('fails closed on an invalid persisted same_round_think_text_policy override before any provider call', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-invalid-persisted-'));
  const storePath = join(dir, 'store');
  const store = JsStore.openOrCreate({ path: storePath });
  try {
    store.registerState({ id: 'framework/state', strategy: 'snapshot' });
  } catch {
    // Already registered.
  }
  store.setStateJson('framework/state', {
    agentRuntimeSettings: {
      agent: {
        sameRoundThinkTextPolicy: 'bogus',
      },
    },
  });
  store.close();

  let providerCalls = 0;
  const rejectingMembrane = {
    complete: async () => {
      providerCalls++;
      throw new Error('provider should not be called');
    },
    streamYielding: () => {
      providerCalls++;
      throw new Error('provider should not be called');
    },
  } as unknown as import('@animalabs/membrane').Membrane;

  try {
    await assert.rejects(
      () => AgentFramework.create({
        storePath,
        membrane: rejectingMembrane,
        agents: [{
          name: 'agent',
          model: 'test-model',
          systemPrompt: 'test',
        }],
        modules: [],
      }),
      /Invalid persisted sameRoundThinkTextPolicy/,
    );
    assert.equal(providerCalls, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
});

it('immediate: a budget decrease with immediate=true applies now — no descent, no persisted flag', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-immediate-'));
  const framework = await AgentFramework.create({
    storePath: join(dir, 'store'),
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      strategy: strategy(),
      contextBudgetTokens: 100_000,
      maxTokens: 10_000,
    }],
    modules: [],
  });
  try {
    // Baseline: a plain decrease starts a paced descent.
    const gradual = framework.updateAgentRuntimeSettings('agent', { contextBudgetTokens: 60_000 });
    assert.equal(gradual.transition, 'converging');
    assert.equal(gradual.contextBudgetTokens, 60_000, 'snapshot reports the TARGET while converging');

    // Immediate: applies now AND cancels the in-flight descent.
    const now = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 50_000,
      immediate: true,
    });
    assert.equal(now.transition, 'stable', 'no descent — the drop is live');
    assert.equal(now.contextBudgetTokens, 50_000);

    // The mode flag is never persisted as an override.
    const overrides = framework.getAgent('agent')!.getRuntimeSettingsOverrides() as Record<string, unknown>;
    assert.equal(overrides.immediate, undefined, 'immediate is a mode, not a setting');
    assert.equal(overrides.contextBudgetTokens, 50_000);
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Budget preflight (issue #122; supersedes feat/budget-preflight-guard).
// The preview stub stands in for the strategy's previewContext so verdicts are
// deterministic; the mapping under test is the framework's, not the picker's.
// ---------------------------------------------------------------------------

function stubPreview(
  framework: AgentFramework,
  agentName: string,
  impl: ((budget: { maxTokens: number }) => Record<string, unknown>) | undefined,
): void {
  // Test-only access to the framework-created manager; Agent exposes no getter.
  const manager = Reflect.get(framework.getAgent(agentName)!, 'contextManager');
  assert.ok(manager instanceof ContextManager);
  const cm = manager as unknown as { previewContext?: (budget: { maxTokens: number }) => unknown };
  if (impl === undefined) {
    cm.previewContext = undefined;
  } else {
    cm.previewContext = (budget) => impl(budget);
  }
}

const infeasibleAt = (floor: number) => (budget: { maxTokens: number }) => ({
  finalTokens: floor,
  budgetTokens: budget.maxTokens,
  fits: floor <= budget.maxTokens,
  exhausted: floor > budget.maxTokens,
  headTokens: 10_000,
  tailTokens: 30_000,
  middleTokens: floor - 40_000,
  middleChunkCount: 7,
  deepestLevel: 3,
  resolutions: {},
  moves: 0,
  producedCount: 0,
});

async function withPreflightFramework(
  fn: (framework: AgentFramework) => Promise<void> | void,
): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-preflight-'));
  const framework = await AgentFramework.create({
    storePath: join(dir, 'store'),
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      strategy: strategy(),
      contextBudgetTokens: 550_000,
      maxTokens: 10_000,
    }],
    modules: [],
  });
  try {
    await fn(framework);
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
}

it('preflight: an immediate infeasible budget throws typed, allowInfeasible overrides', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', infeasibleAt(450_000));
    assert.throws(
      () => framework.updateAgentRuntimeSettings('agent', {
        contextBudgetTokens: 260_000,
        immediate: true,
      }),
      (error: unknown) => {
        assert.ok(error instanceof BudgetPreflightError);
        assert.equal(error.preview.path, 'immediate');
        assert.equal(error.preview.effective?.fits, false);
        assert.equal(error.preview.effective?.finalTokens, 450_000);
        assert.match(error.message, /450000 tokens against hard budget 260000/);
        return true;
      },
    );
    // The refused patch must not have been applied.
    assert.equal(framework.getAgentRuntimeSettings('agent').contextBudgetTokens, 550_000);

    const applied = framework.updateAgentRuntimeSettings(
      'agent',
      { contextBudgetTokens: 260_000, immediate: true },
      { allowInfeasible: true },
    );
    assert.equal(applied.contextBudgetTokens, 260_000);
  });
});

it('preflight: a paced descent below the floor NEVER blocks — the compile budget is unchanged', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', infeasibleAt(450_000));
    // 260k < 550k live, no `immediate` → paced. The superseded branch guard
    // previewed 260k as the compile budget and refused this exact patch; the
    // real compile stays at 550k (feasible), so it must apply.
    const applied = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 260_000,
    });
    assert.equal(applied.transition, 'converging');
    assert.equal(applied.contextBudgetTokens, 260_000, 'snapshot reports the target');
  });
});

it('preflight: a feasible immediate change applies without ceremony', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', infeasibleAt(200_000));
    const applied = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 260_000,
      immediate: true,
    });
    assert.equal(applied.contextBudgetTokens, 260_000);
  });
});

it('preflight: preview-unavailable applies with a warn, never blocks', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', undefined);
    const applied = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 260_000,
      immediate: true,
    });
    assert.equal(applied.contextBudgetTokens, 260_000);
  });
});

it('previewAgentRuntimeSettings: structured reasons for the strategy throw paths', async () => {
  await withPreflightFramework(async (framework) => {
    const cases: Array<[string, string]> = [
      [
        'AutobiographicalStrategy.previewContext requires reinitialization for the current branch generation',
        'branch_generation_changed',
      ],
      ['previewContext is already running; previews must not overlap', 'preview_in_flight'],
      [
        'previewContext requires adaptiveResolution; the hierarchical path has no fold plan to preview',
        'no_adaptive_resolution',
      ],
      ['some novel failure', 'some novel failure'],
    ];
    for (const [message, reason] of cases) {
      stubPreview(framework, 'agent', () => { throw new Error(message); });
      const preview = framework.previewAgentRuntimeSettings('agent', {
        contextBudgetTokens: 600_000,
      });
      assert.equal(preview.available, false, message);
      assert.equal(preview.reason, reason);
    }
  });
});

it('previewAgentRuntimeSettings: paced patches report effective=live plus an advisory target', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', infeasibleAt(450_000));
    const preview = framework.previewAgentRuntimeSettings('agent', {
      contextBudgetTokens: 260_000,
    });
    assert.equal(preview.path, 'paced');
    assert.equal(preview.available, true);
    assert.equal(preview.effective?.budgetTokens, 550_000, 'effective = UNCHANGED live budget');
    assert.equal(preview.effective?.fits, true);
    assert.equal(preview.advisory?.targetTokens, 260_000);
    assert.equal(preview.advisory?.fits, false);

    // No patch = the resume-gate case: verdict for current settings.
    const current = framework.previewAgentRuntimeSettings('agent');
    assert.equal(current.path, 'none');
    assert.equal(current.effective?.budgetTokens, 550_000);
  });
});

it('preflight: boot restore is never preflighted — an infeasible persisted budget cannot brick startup', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-settings-preflight-boot-'));
  const storePath = join(dir, 'store');
  const config = () => ({
    storePath,
    membrane,
    agents: [{
      name: 'agent',
      model: 'test-model',
      systemPrompt: 'test',
      strategy: strategy(),
      contextBudgetTokens: 550_000,
      maxTokens: 10_000,
    }],
    modules: [],
  });
  let framework = await AgentFramework.create(config());
  try {
    stubPreview(framework, 'agent', infeasibleAt(450_000));
    framework.updateAgentRuntimeSettings(
      'agent',
      { contextBudgetTokens: 260_000, immediate: true },
      { allowInfeasible: true },
    );
    await framework.stop();
    // Recreate WITHOUT a stub: restore goes through Agent.restoreRuntimeSettings,
    // bypassing the framework wrapper — must not throw regardless of feasibility.
    framework = await AgentFramework.create(config());
    assert.equal(framework.getAgentRuntimeSettings('agent').contextBudgetTokens, 260_000);
  } finally {
    await framework.stop();
    rmSync(dir, { recursive: true, force: true });
  }
});

it('preflight: no-op and increasing patches never throw on an already-over-floor agent', async () => {
  await withPreflightFramework(async (framework) => {
    stubPreview(framework, 'agent', infeasibleAt(600_000)); // floor above even 550k live
    // No-op rewrite of the current budget: settings-plane writeback shape.
    const rewrite = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 550_000,
      immediate: true,
    });
    assert.equal(rewrite.contextBudgetTokens, 550_000);
    // Strict improvement (increase) on a wedged agent must apply, not throw.
    const raise = framework.updateAgentRuntimeSettings('agent', {
      contextBudgetTokens: 700_000,
    });
    assert.equal(raise.contextBudgetTokens, 700_000);
    // A LOWERING that does not fit still throws.
    assert.throws(
      () => framework.updateAgentRuntimeSettings('agent', {
        contextBudgetTokens: 500_000,
        immediate: true,
      }),
      BudgetPreflightError,
    );
  });
});
