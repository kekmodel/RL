# Banking Environment Spec (as-built, v0)

This is an as-built specification for the `nemo_rl.environments.banking`
package. It describes the system as it exists at the commit that adds
this file. When the upstream kakaobank task schema changes, update this
spec first, then drive the implementation changes from the spec diff.

## Goal

Provide a binary (0/1) reward for RL training on the kakaobank_manual_v0
task corpus, decidable purely by comparing the canonical hash of a
replayed gold DB state against the canonical hash of the agent's
trajectory-induced DB state.

## Scope

**In scope:**
- The 203 tasks in `data/tasks/kakaobank_manual_v0/` whose
  `reward_basis` includes `"DB"` and whose `expected_actions` contain no
  entries with `requestor == "user"`.

**Out of scope (deferred):**
- The 4 user-tool tasks (require a user simulator).
- `COMMUNICATE` / `NL_ASSERTION` evaluation (needs an LLM judge).
- The 14 `verifier_assertion` types are not evaluated independently;
  they are subsumed by canonical-hash equality when they hold against a
  gold-replayed reference state.
- Dataset processor, prompt construction, curriculum selection, and
  registration in `ENV_REGISTRY` / `ACTOR_ENVIRONMENT_REGISTRY`.

## Module Layout

```
nemo_rl/environments/banking/
├── state.py        — KakaoBankState, 26 tables, canonical_hash
├── task_loader.py  — JSON loader + DB-decidability filter
├── tools.py        — apply_action, 19 action-family handlers
├── runner.py       — BankingRunner turn logic + Qwen3-coder parser
└── environment.py  — @ray.remote BankingEnvironment (Ray actor)
```

## Inputs: Task JSON Schema

Source files: `kb_manual_*.json` under the kabang-knowledge repo's
`data/tasks/kakaobank_manual_v0/` directory.

Fields consumed by this package:

| Field | Type | Purpose |
|---|---|---|
| `task_id` | str | Identity (metrics, logging) |
| `product_names` | list[str] | Optional filter key for `load_tasks` |
| `task_type` | str | `"success"`, `"refusal"`, `"adversarial"`, `"user_tool"` |
| `user_prompt` | str | First user turn |
| `initial_state` | dict[table, {"data": {id: record}}] | Seed for predicted and gold state |
| `expected_actions` | list[{name, requestor, arguments}] | Gold replay script |
| `reward_basis` | list[str] | Must include `"DB"` to pass the decidability filter |
| `required_documents` | list[str] | KB oracle IDs; not consumed by the DB-hash reward path |

Filter (`task_loader.is_db_decidable`):

1. `"DB" in reward_basis`
2. No `expected_actions` entry has `requestor == "user"`

## State Model

`KakaoBankState = dict[table_name, {"data": dict[record_id, record_body]}]`.

The 26 canonical tables, in order, are enumerated in
`state.KAKAOBANK_TABLES`:

`customers`, `businesses`, `consents`, `accounts`, `deposit_contracts`,
`savings_boxes`, `auto_transfer_rules`, `group_memberships`, `pockets`,
`child_relationships`, `cards`, `card_orders`, `prepaid_wallets`,
`loans`, `loan_applications`, `refinance_requests`,
`required_documents`, `mortgage_collateral`, `lease_contracts`,
`vehicle_purchase_cases`, `comparison_sessions`, `remittance_profiles`,
`remittance_cases`, `service_enrollments`, `transactions`, `disputes`.

Seeding (`state.seed_state`):

- Base is `empty_state()`: every table present with `{"data": {}}`.
- For each table in `initial_state`, record bodies are deep-copied in.
- Tables not present in `initial_state` remain empty.

Canonical hash (`state.canonical_hash`):

- `sha256(json.dumps(state, sort_keys=True, ensure_ascii=False,
  default=str, separators=(",", ":")).encode("utf-8")).hexdigest()`
- Stable across runs and independent of dict insertion order.
- Non-JSON-native scalars (e.g. `datetime.date`) are coerced by
  `default=str`.

## Action Dispatcher

`tools.apply_action(state, action)` dispatches on `action["name"]` and
mutates `state` in place. Unknown names are no-ops; `arguments` that is
missing or `None` is treated as `{}`.

### Read tools (state-invariant)

`KB_search`, `get_customer_profile`, `get_account_or_contract`.

### Write tools

Each handler below is deterministic in `(name, arguments)`, idempotent,
and no-ops when its primary id argument is absent.

| Handler | Table(s) mutated | Primary id resolution | Status / shape it produces |
|---|---|---|---|
| `close_account_or_service` | first match in `(accounts, deposit_contracts, savings_boxes, group_memberships, pockets, service_enrollments, prepaid_wallets)` | `target_id` | `status=CLOSED`, optional `close_type`, `close_reason` |
| `open_or_enroll_product` | caller-specified `target_table` | `target_id` | record with `customer_id`, `product_name`, `status=ACTIVE` (default); merges caller `options` |
| `update_card_state` | `card_orders`; optional cross-write into `cards` for `BLOCK` / `RESTRICT` / `LOST_REPORT` / reissue-approval | `order_id` or `card_order_id` | order status from `operation` (`REJECT_NEW_ISSUE→REJECTED`, `APPROVE_*/ISSUE/REISSUE_CARD→APPROVED`, `CANCEL→CANCELLED`) |
| `update_loan_contract_state` | `loans` | `loan_id` or `target_id` | status from `operation` (`ACCELERATE*→ACCELERATED`, `EXECUTE→EXECUTED`, `WITHDRAW→WITHDRAWN`, etc.) |
| `process_refinance_request` | `refinance_requests` | `refinance_id` | status from operation prefix (`CANCEL*→CANCELLED`, `COMPLETE*→COMPLETED`) |
| `request_maturity_or_extension` | first match in `(deposit_contracts, loans)` | `target_id` | records `maturity_decision`; `MATURE*→status=CLOSED`, `AUTO_CLOSE*→AUTO close`, `EXTEND*→EXTENDED`, `REJECT*` preserves original status |
| `execute_remittance_case` | `remittance_cases` | `remittance_id` (top-level) or `options.remittance_id` | case record with direction, amount, currency, country, purpose_code |
| `execute_deposit_or_box_transfer` | `transactions` | explicit `transaction_id` or deterministic composite `txn::{type}::{source}::{target}::{amount}::{currency}` | status `REJECTED` when `transaction_type` starts with `REJECT`, else `POSTED` |
| `configure_auto_transfer` | `auto_transfer_rules` | `auto_transfer_id` / `existing_auto_transfer_id` / `options.auto_transfer_id` | status from operation (`CANCEL→CANCELLED`, `REJECT*→REJECTED`, default `ACTIVE`) |
| `request_interest_payment` | `transactions` | composite `interest::{target_id}::{interest_amount_krw}` | `transaction_type=INTEREST_PAYMENT`, `status=POSTED` |
| `create_loan_application` | `loan_applications` | `application_id` | records amount, purpose, partner; `status` from `expected_status` or `SUBMITTED` default |
| `file_dispute_or_objection` | `disputes` | `dispute_id` or `dispute::{customer_id}::{target_id}` | `status=FILED`, records reason and target_type |

Action families **not** implemented (unused in the DB-decidable subset
of kakaobank_manual_v0): `log_identity_verification`,
`customer_apply_for_product`, `submit_required_document`,
`request_human_transfer`.

## Wire Format (Nemotron 3 Nano / Qwen3-coder)

**Agent → environment** (model-emitted `<tool_call>` block parsed by
`runner._parse_action`):

```
<tool_call>
<function=tool_name>
<parameter=key>value</parameter>
...
</function>
</tool_call>
```

- Parameter values are text; coerced via `json.loads` → `ast.literal_eval`
  → trimmed string fallback (the Qwen3-coder no-schema path).
- When multiple `<tool_call>` blocks appear, the LAST one wins
  (accommodates self-correction).
- A `<tool_call>` lacking a well-formed `<function=NAME>...</function>`
  body is flagged as malformed and increments `num_turns` without
  terminating.

**Environment → agent** (tool response):

- Returned as `{"role": "tool", "content": <plain>}`.
- The tokenizer's chat template wraps this as
  `<|im_start|>user\n<tool_response>\n{content}\n</tool_response>...<|im_end|>`.
  The runner never emits those tags itself.

**Stop string:** `</tool_call>`.

**Thinking tokens** (chat template owns this, runner is oblivious):

- Default `enable_thinking=True`: the generation prompt ends with
  `<|im_start|>assistant\n<think>\n`, opening a fresh reasoning section
  for every next assistant turn.
- Default `truncate_history_thinking=True`: `reasoning_content` on
  assistant turns that predate the LAST user turn is stripped; the
  corresponding `<tool_call>` XML is preserved so the DB-state trace
  stays visible.
- Contiguous `role: "tool"` messages are batched under one `user` turn
  by the template.

## Reward & Termination

Reward is computed as:

```
reward = 1.0 if canonical_hash(predicted_state) == gold_hash else 0.0
```

where `gold_hash = canonical_hash(apply_actions(seed_state(initial_state),
expected_actions))` is precomputed once per task.

`BankingRunner.process_turn` terminates the episode on any of:

1. The just-applied tool call makes `canonical_hash(predicted_state)`
   equal `gold_hash` → reward 1.0.
2. The assistant message contains no `<tool_call>` block → reward is
   the hash comparison against the current predicted state.
3. `num_turns >= max_turns` → same hash comparison.
4. `__malformed__` tool call → **not** terminal; increments num_turns
   and returns an explanatory tool-role observation.

## Invariants

- **Determinism:** `apply_action(state, action)` depends only on `state`
  and `action`. No randomness, clock reads, or I/O.
- **Idempotency:** every write handler can be applied twice with
  identical args without changing the canonical hash after the first
  application.
- **Safe on missing primary id:** every write handler is a no-op when
  its primary id arg is missing; canonical hash is preserved exactly.
- **Caller-metadata immutability:** `BankingRunner.process_turn` never
  mutates the input `metadata` dict; returned `next_metadata` carries a
  deep-copied `predicted_state`.
- **Reward monotone in hash equality:** reward is 1.0 iff hashes match.

## Test Surfaces

Unit specs live in `tests/unit/environments/test_banking_*.py`.
Integration tests that render through the real Nemotron 3 Nano
tokenizer are gated on the `NEMOTRON_TOKENIZER_DIR` environment variable
and are skipped by default.

## Open Questions (re-visit on kakaobank schema updates)

- **`initial_state` shape stability**: the nested
  `{table: {"data": {record_id: body}}}` layout is assumed by
  `seed_state`; a flatter layout would require a loader change.
- **Argument key stability**: the write-tool handlers rely on specific
  arg names (`target_id`, `refinance_id`, `operation`, `options`, etc.).
  Renames in upstream task JSONs propagate directly into handler
  updates.
- **New `operation` strings**: `update_card_state`,
  `update_loan_contract_state`, `process_refinance_request`, and
  `request_maturity_or_extension` dispatch on a string that is not
  constrained by any upstream schema. New operations need explicit
  status-mapping entries.
- **User-tool coverage**: if the upstream adds a user simulator, the
  `requestor == "user"` exclusion in `is_db_decidable` can be relaxed
  and the corresponding tasks brought in-scope.
- **Reward-basis expansion**: tasks that currently rely on
  `COMMUNICATE` or `NL_ASSERTION` for primary signal are out of scope;
  adding them would require a judge-backed environment, not a
  DB-hash-only one.
