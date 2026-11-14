# Project Progress Tracker

Updated after Phase 7.4 classifier bundle integration. “Complete” here means the stated code/tests exist; sample-backed behavior is not a live ecommerce integration.

| Phase | Status | Evidence / remaining work |
| --- | --- | --- |
| 0 — Workspace setup | Complete | Independent `ecom_langraph/`; original classifier/reference projects remain untouched. |
| 1 — Flask/LangGraph foundation | Complete | Flask factory, health endpoint, basic graph and tests. |
| 2 — XLM-R + Groq | Complete | Lazy local classifier, confidence-gated verifier, actual 15 labels, fail-closed behavior. |
| 2.1 — Self-contained classifier bundle | Complete (local bundle; publication review pending) | Inference weights/tokenizer and verifier descriptions are inside `ecom_langraph`; default loading no longer depends on the sibling classifier folder. Weight file is configured for Git LFS. |
| 3 — Intent orchestration | Complete | Typed state, classifier-first conditional graph, `InMemorySaver`; payment/shipping routes remain pending/safe. |
| 4A — Business analysis/docs | Complete | Reference functions and risks inspected; permanent README/system design established. |
| 4B — Order/product workflows | Complete (sample data) | JSON repository, services, tools, graph integration, tests; no live backend. |
| 4C — Core support workflows | Complete (simulation) | Issue resolver, return draft, simulated damage/missing cases and handoff, store information, ETA delay assessment; no real external operations. |
| 4D — Conversational continuity | Complete (process-local) | Pending slots, resume/interruption, session isolation, confirmation gate; InMemorySaver only. |
| 4E.1 - PostgreSQL foundation | Complete (optional/lazy) | SQLAlchemy/psycopg engine/session, Alembic baseline, URL/config tests. |
| 4E.2 - Business data migration | Complete (opt-in) | Five PostgreSQL models, repositories, explicit idempotent JSON importer; JSON remains default and live PG validation is pending. |
| 4E.3 - LangGraph PostgreSQL checkpointer | Complete (opt-in) | Configurable memory/PostgreSQL saver; pooled process lifecycle; real two-process resume/isolation test passed. Authentication and retention controls remain outstanding. |
| 5.1 — RAG integration plan | Complete (planning only) | Inspected current graph/store workflow and reference RAG implementation; documented minimal design, incompatibilities, config, dependencies, and offline tests. No RAG implementation or provider dependency added. |
| 5.2 — Document RAG foundation | Complete (basic) | Lazy service and offline Markdown/text indexer reused by Phase 5.4. No approved documents, Pinecone index, or live provider tests yet. |
| 5.3 — Agentic RAG architecture analysis | Complete (documentation only) | Inspected reference controller/retriever/augmenter/generator/tools/API/evaluation and current graph; documented actual gaps and proposed bounded LangGraph subgraph. No code changed. |
| 5.4 — Agentic RAG implementation | Complete (bounded) | LangGraph subgraph on `store_information`; structured evidence evaluation, one rewrite/retrieval retry maximum, grounded answer or safe fallback; fake-client tests. |
| 5.5 — Knowledge base and RAG quality validation | Complete (offline) | Metadata schema, approved/demo separation, content-versioned IDs, prefix/namespace-bounded stale cleanup, fictional test corpus and offline report. No live Pinecone/provider evaluation. |
| 6.1 — Flask chat API | Complete (unauthenticated) | `POST /api/chat` invokes existing graph, validates payload/size, projects safe fields, allowlist CORS; health unchanged. Session ID remains caller-controlled and is not authentication. |
| 6.2 — Optional FakeStore + conversational testing | Complete (demo catalog) | Lazy stdlib HTTP client; bounded product listing/search, explicit-ID details, no stock quantities; optional product tools; PowerShell API scenario runner. |
| 6.3 — Conversational routing + refund demonstrations | Complete (offline validated) | Classifies new requests; valid pending slots skip classifier; invalid slots invoke it once for interruption detection; verified interruptions preserve/restore prior workflow; demo refund states/import tests; damage conversation tested through Flask API. Full suite: 182 passed, 2 skipped. |
| 6.3.1 — Policy routing + FakeStore source isolation | Complete (offline validated) | Explicit return/refund policy questions refine to existing RAG route without changing classifier labels; FakeStore misses/failures no longer fall through to local product data. |
| 6.4 — Customer-facing frontend | Superseded | Frontend direction moved to the independent Next.js project in Phase 7.1. |
| 7.1 — Next.js frontend foundation | Complete | App Router, React, TypeScript, Tailwind, responsive layout, and persistent system-aware themes. |
| 7.2 — Interactive chat UI | Complete (local mock) | Initial greeting, multi-turn message bubbles, typing indicator, Enter/Shift+Enter, and auto-scroll; no Flask calls or persistence. |
| 7.3 — Flask frontend integration | Complete (client implemented) | Configurable API base URL, stable per-conversation UUID, backend answers and error handling. Live Flask requests were not tested because no server was available. |
| 7.4 — Standalone classifier packaging | Complete (offline validated) | Bundled model artifacts under `models/xlmr_intent_classifier`; lazy local loading and clear missing-file errors; original classifier remains unchanged. Verify fine-tuned model/data redistribution rights before public release. |
| 7 — Deployment | Planned | Production security, monitoring, evaluation and deployment hardening. |

## Phase 7.4 implementation evidence

- The actual sibling folder found in this workspace is `xlmr_openai_intent_verifier`; the user-specified `xlmr_openai_intent_verification` directory does not exist. No original classifier or reference-project files were modified.
- Runtime imports remain app-owned: `app.classifier.classifier_service` and `app.classifier.model_loader`. They do not import the source project's `main.py` or `routing.py`.
- Copied only inference inputs: `config.json`, `label_mapping.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json` into `models/xlmr_intent_classifier/`; copied `intent_description.json` into `app/classifier/intent_descriptions.json` for the existing conditional verifier. Training arguments, training code, data, archives, `.env`, and virtual environments were excluded.
- The local model directory is the default, `local_files_only=True` remains in use, model caching remains lazy, and missing core files now produce an actionable error. The default 95.0 threshold and original label map remain unchanged.
- `model.safetensors` is 1,112,244,996 bytes (about 1.04 GiB); `.gitattributes` routes it through Git LFS. GitHub's normal Git file cap is 100 MiB; Git LFS is needed for a complete clone. Review the current account storage/bandwidth plan limits before publication.
- The original project contains no project-level model license or training-data redistribution terms. The upstream XLM-R base model is listed as MIT, but that does not establish this fine-tuned artifact's terms. Public redistribution remains a rights-review item.
- Validation: classifier suite passed 16 tests; the full backend suite ran **197 tests: 195 passed and 2 skipped**. The actual bundled model loaded on CPU and `classify_and_verify` predicted `order_tracking` for “Where is my order?” (99.57%) and “मेरा ऑर्डर कहाँ पहुँचा है?” (96.82%), accepting both above the unchanged threshold without calling Groq. Existing mocked tests cover low-confidence Groq success/correction and failure behavior.
- SHA-256 comparisons confirmed the bundled model weights, tokenizer files, config, label map, and verifier descriptions are byte-identical to the source artifacts. No original project, reference project, Rasa feature, or business workflow was changed.

## Phase 7.2 implementation evidence

- Kept the existing Next.js page, sidebar, responsive styles and persisted light/dark theme. The page heading remains “E-commerce Customer Support”; every new page load begins with the assistant message “Hi! How can we help you today?”.
- Added local multi-message state with right-aligned customer bubbles and left-aligned assistant bubbles. Sending by Enter or the Send button clears the input; Shift+Enter inserts a newline. Empty/whitespace-only messages are ignored.
- A brief typing indicator precedes each local mock assistant reply. The composer remains available while replies are pending, and the scroll view follows the newest message.
- No Flask/API requests, authentication, backend changes, or conversation persistence were added. A page refresh resets the conversation to the initial greeting.
- Validation: `npm run build` and `npm run lint` passed. Replies are local mock text and do not reflect classifier, RAG, or business-service results.

## Phase 7.3 implementation evidence

- Inspected `app/api/routes.py`: `POST /api/chat` accepts only non-empty `session_id` and `message`, enforces configured length limits, and returns `session_id`, `answer`, `intent`, and `workflow_status`. Error bodies return `error` and a safe `message`; graph failures return HTTP 503. The backend echoes the normalized session ID.
- Replaced mock responses with browser `fetch` requests to `${NEXT_PUBLIC_CHAT_API_BASE_URL}/api/chat`. The client reads `answer`, verifies the echoed session ID, and displays safe API/network/configuration errors without inventing an answer.
- Generates a browser UUID on the first turn and reuses it for later turns. **New chat** aborts any active request, clears local UI state and starts a new conversation ID on the next send. The ID is only a conversation key, not authentication or customer identity.
- The frontend environment file is `frontend/.env.local`, copied from `frontend/.env.example`; it contains only the public API base URL. Flask must allow the frontend's exact origin in `CORS_ALLOWED_ORIGINS`.
- Chat history and session ID remain in React memory and are lost on refresh. No backend, classifier, graph, database, or RAG code changed.
- Validation: `npm run build` and `npm run lint` passed. A safe health check found no Flask server listening on `127.0.0.1:5000`, so consecutive live API turns were not tested. Manual instructions are in the root README.

## Phase 6.3 implementation evidence

- No pending workflow: every new customer request reaches the existing XLM-R/Groq classifier node. Existing model, label map, and 95% verifier threshold are unchanged.
- Pending workflow: continuity validates the active slot first. Valid order IDs, product names, issue choices, confirmation answers, and adequately formed free-text details resume directly without classifier calls. Invalid slot messages invoke the classifier exactly once. A verified different request is routed as an interruption; unverified/unknown output asks for clarification and leaves the pending state intact.
- Interrupted workflows are held in `suspended_workflows` in the existing LangGraph checkpoint state. A terminal `finalize_turn_node` restores them after the current request completes. Independent thread IDs continue to isolate these lists.
- The damage sequence from the request is tested through Flask's test client and actual LangGraph; only the first turn calls the classifier. The final turn records one local demo support case after the existing order service validates `ORD-1001`; it states no retailer/courier claim was submitted.
- Refund fixture states: `ORD-1001` = `not_initiated`, `ORD-1002` = `processing`, `ORD-1003` = `completed`; `ORD-1004` has no refund row. A fourth fictional order was added to cover the absent-record case. Existing refund schema already represented these states, so no migration was needed.
- The fixture importer still skips existing IDs and never updates them. Refund import now also skips if the referenced order is absent or not marked as demo, preventing demo status rows from attaching to non-demo orders.
- `scripts/run_chat_scenarios.ps1` covers requested API scenarios, with unique IDs per scenario and shared IDs for multi-turn messages. Syntax is checked; live Flask-server execution is not claimed.
- Relevant validation: continuity, API, workflow, model/import, and repository tests passed. Full suite: 182 passed, 2 skipped. Import dry run reports 3 products, 4 orders, 3 refunds with zero errors; idempotency is checked using in-memory session doubles, not a live PostgreSQL server.
- Classifier-call tests cover no-pending new requests, valid slot responses, invalid slot responses, verified interruption, and unverified interruption. Hindi yes confirmation and a Hindi damage-description slot are also covered.

## Phase 6.3.1 implementation evidence

- `route_classifier_result()` now sends only explicit general return/refund policy questions from a verified `issue_support` result to the existing `store_information` route. The state retains the classifier's original prediction and final intent. Action requests and refund-status questions remain on the issue resolver.
- No approved policy documents are present in `knowledge_base/approved/`. Demo policy files remain segregated and are not promoted or treated as official. Missing RAG evidence/provider configuration returns the existing safe unavailable response.
- `FAKESTORE_ENABLED=false` uses the configured local demo product repository directly. When enabled, FakeStore success, not-found and unavailable outcomes are terminal for product lookup; product-name slot validation also uses FakeStore and does not query the local repository. FakeStore responses never supply inventory quantities.
- Focused tests cover policy routes versus return/refund transactions, FakeStore disabled/local selection, successful catalogue results, not-found, provider failure, unknown stock, generic availability clarification, explicit product 404 and pending-slot source isolation.
- Validation: `.venv` Python full unittest suite passed **192 tests; 2 skipped**. No paid provider call or live FakeStore request was made.

## Phase 6.2 implementation evidence

- Reference source inspection: `api/integrations/fakestore_client.py` performs a plain `requests.get` to `/products/{id}` and immediately calls `.json()` without timeout, HTTP status handling or schema validation. Reference `get_order_status` and `get_product_stock` read local fixture dictionaries; the FakeStore client is not used by those services. The README's “real-time” order status and `check_stock` descriptions therefore overstate what the inspected source implements; its tool registry names `check_product_stock` and does not call FakeStore.
- The new optional `app/integrations/fakestore_client.py` uses only Python's standard library, validates product object/list schemas, sets a request timeout and gives sanitized failures. It calls `GET /products?limit=N` and `GET /products/{id}` only.
- `app/tools/fakestore_tools.py` provides list, search, detail, and stock inquiry wrappers. The stock wrapper always returns `stock_quantity: None`; FakeStore product records have no inventory count. There are no cart writes. Although the public API documents cart endpoints, the current project has no customer cart identity/state lifecycle to safely connect them.
- The existing classifier routes `product_inquiry` to its node; only that node optionally selects the FakeStore catalog when `FAKESTORE_ENABLED=true`. The other labels and routing remain unchanged. Phase 6.3.1 now makes FakeStore misses and failures terminal when it is selected; when disabled, local demonstration product data is selected directly. Order and support paths remain independent.
- Order tracking continues to call the existing tool -> service -> repository. With `BUSINESS_DATA_BACKEND=postgres`, that repository uses the Phase 4E.2 PostgreSQL implementation; local default remains demonstration JSON. The project makes no FakeStore order request.
- `scripts/run_chat_scenarios.ps1` sends real Flask HTTP requests sequentially, using a separate session ID per scenario and the same ID for all turns inside it. Unit/API integration tests use fake HTTP responses and do not call external providers.
- Validation: `python -m unittest tests.test_fakestore -v` passed 6 tests, including an actual Flask test-client request traversing LangGraph with a mocked FakeStore HTTP response. `.\.venv\Scripts\python.exe -m unittest discover -s tests -q` passed **174 tests; 2 skipped**. The skipped tests are optional integrations; no live FakeStore or paid-provider call was made.

## Phase 4B implementation evidence

- `data/mock/orders.json` and `products.json` are explicitly marked demonstration-only and contain no customer email addresses.
- `app/services/data_repository.py` isolates JSON access from graph code.
- `app/services/order_service.py` validates order IDs and exposes only required order fields.
- `app/services/product_service.py` searches by product ID/name, returns ambiguity explicitly, and does not invent stock.
- `app/tools/order_tools.py` and `product_tools.py` wrap these services without touching graph state.
- `app/graph/nodes.py` now performs sample-backed order tracking and product stock inquiry for the existing trained intents.
- `tests/test_order_service.py`, `test_product_service.py`, and `test_business_graph.py` cover service outcomes, user-facing branches, and that unrelated routes do not invoke the tools.
- Complete test run: 53 tests passed, one opt-in real-classifier smoke test skipped.

## Current limitations

- JSON fixture values and dates are examples, not live orders or current inventory.
- At Phase 4D, checkpoints were process-local. Phase 4E.3 adds optional PostgreSQL persistence; session IDs still isolate thread namespaces only and do not authenticate callers.
- The Flask chat endpoint is unauthenticated; no Streamlit UI, live service authorization, or production database deployment exists. Demo case records use local SQLite.
- Payment/shipping integrations, RAG routing outside `store_information`, approved RAG corpus/index, real return/refund operations, and mutations remain unimplemented; Phase 4C support workflows are local simulations.

## Phase 6.1 implementation evidence

- Added `POST /api/chat` to the existing `app/api/routes.py` blueprint; preserved `GET /api/health` response and reused `invoke_graph(message, session_id)` rather than duplicating graph or business logic.
- Request validation requires JSON object fields `session_id` and `message`, rejects extra/internal fields, malformed JSON, blank/wrong-typed values, messages over `MAX_CHAT_MESSAGE_LENGTH`, IDs over `MAX_SESSION_ID_LENGTH`, and request bodies over `MAX_CONTENT_LENGTH`.
- Public success response maps actual graph fields to `session_id`, `answer` (`response`), `intent` (`final_intent`) and `workflow_status`; other graph state, messages, tools, and errors are not exposed. Graph exceptions produce a generic 503 response.
- Added environment-configured exact-origin CORS through `CORS_ALLOWED_ORIGINS`; wildcard is rejected, preflight supports JSON, and credentials are not allowed. CORS does not authenticate a caller.
- API tests cover success, validation, size limits, safe graph errors, session continuity/isolation, health, and CORS. Full suite `python -m unittest discover -s tests -q`: **168 tests passed, 2 skipped**. The integration test uses an in-memory saver and mocked classifier. No external API calls were made.
- No authentication, user ownership binding, authorization, rate limiting, or session retention/deletion policy was added. Client-chosen session IDs must not be exposed as secure identities.

## Next phase gates

1. Phases 4C and 4D are complete as demonstration implementations; do not treat local support records as real service actions.
2. Phases 4E.2 business migration and 4E.3 checkpoint persistence are complete and opt-in; authenticated ownership and retention controls remain future work.
3. Keep sample-data labels visible in responses and never imply live freshness.
4. Do not implement mutations until real backend contracts, authorization, idempotency and explicit user confirmation are defined.
5. Update README and system design alongside implementation evidence at each phase boundary.


## Phase 5.1 inspection evidence

- Current `store_information_node` calls `store_information_tool` and reads fictional structured data from `data/mock/store_information.json`. The graph's RAG node remains a placeholder; routing does not use a new classifier label.
- Inspected the reference loader, chunker, vector store, retriever, augmenter, generator, controller, tool registry, indexing script, route initialization and requirements. Useful concepts and concrete incompatibilities are documented in `docs/system_design.md`, section 23.
- The reference's eager Pinecone/OpenAI initialization, inconsistent index names/document result shape, undeclared imports, Rasa-era tool imports, and unreliable API-dependent tests make it unsuitable as a runtime dependency or wholesale code copy.
- Planned design: approved Markdown/JSON docs -> explicit offline chunk/embed/index command -> one Pinecone index -> lazy query embedding/retrieval -> grounded DeepSeek generation with source identifiers and insufficient-evidence fallback. Preserve deterministic order/refund/return operations.
- Planned provider variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `DEEPSEEK_API_KEY`; planned optional settings include index/model/top-k/score thresholds. The current app does not read them. Planned minimal added SDKs: `openai` and `pinecone`; no packages were installed or added in this phase.
- Ordinary RAG tests will use fake provider clients and cover metadata, retrieval filtering, citations, empty/weak evidence, provider failures, lazy initialization and business-route isolation. Real provider smoke tests remain opt-in.
- Phase 5.1 changed documentation only. No RAG code, provider calls, index, or document ingestion were added. Phase 5.2 has not started.


## Phase 4C implementation evidence (historical)

- Existing classifier label set is unchanged. `app/graph/routing.py` maps the existing `issue_support`, `customer_care_support`, and `store_information` labels to their new nodes.
- `app/services/issue_resolver.py` routes four supported issue subtypes and asks for clarification when ambiguous; tests assert the ambiguous case executes no issue tools.
- At Phase 4C the refund fixture was empty; Phase 6.3 supersedes that state with the three clearly labeled demo statuses and a separate order that has no record. Missing records still never imply a refund status.
- Return policy/eligibility is read from demo JSON. Missing purchase date and item condition result in manual review. Return requests are local SQLite demo drafts only.
- Damage/missing reports and customer handoffs create clearly marked local demo records. No actual merchant/courier claim or human contact occurs; missing packages are not declared lost.
- Fictional store information and promotions are read from structured demo JSON and stated as not live/redeemable.
- Order delay is derived from the fixture status and ETA only; unknown status is not treated as a tracking event.
- `data/runtime/` is Git-ignored. Tests use temporary SQLite paths and do not depend on live services, Pinecone, DeepSeek, Rasa, Groq, or classifier weights.
- Phase 4C had no reliable cross-turn slot resumption; Phase 4D implements it. PostgreSQL persistence remains Phase 4E.

## Phase 4C test evidence

Full local test command: ` .\.venv\Scripts\python.exe -m unittest discover -s tests -v `. Phase 4C historical run: 70 tests ran, 69 passed, and one optional local-classifier smoke test was skipped.

## Limitations at the end of Phase 4C

- Every business fixture and support record is demonstration data, not live customer information or operational data.
- Refund status has no fixture records. Return drafts require separate manual review; damage/missing reports and handoff are simulated only.
- At the end of Phase 4C, missing required details could not reliably resume across turns; Phase 4D implements same-thread continuation.
- No chat API, Streamlit frontend, live service authorization, production database, RAG, or live payment/shipping integration exists.

## Next phase gates

1. Phase 4D is implemented; PostgreSQL checkpointer migration remains Phase 4E.3.
2. Phase 5.2 provides the basic RAG provider/indexing foundation; Phase 5.4 adds bounded LangGraph orchestration. Add approved documents and configure Pinecone before expecting answers, then evaluate retrieval/evidence quality against that corpus.
3. Phase 6: expose conversation endpoint before implementing Streamlit.
4. Keep labels fixed and demo-data boundaries visible; do not implement real mutations without service contracts, authorization, idempotency and explicit confirmation.


## Phase 4D implementation evidence

- `app/graph/state.py` now stores an explicit pending workflow, required/collected slots, clarification prompt, workflow status, resumption flag, and confirmation state.
- `app/graph/workflow.py` routes every turn through `continuity_node` before classifier routing. `build_graph(checkpointer=None)` preserves the injectable LangGraph checkpointer boundary; local default remains `InMemorySaver`.
- Strict order IDs, product lookup/ambiguity, and explicit issue subtype answers validate against the expected slot. Free-text issue details resume only when the existing classifier accepts the expected `issue_support` intent.
- A different classified request can run while the previous workflow is suspended; a later valid answer resumes that workflow in the same thread. One pending workflow is active per thread; a second incomplete workflow can replace it.
- Return, damage, and missing-delivery workflows collect needed details across turns. Missing-delivery demo records include customer-provided detail and still do not claim a parcel is lost or a carrier was contacted.
- Confirmation-pending state requires explicit confirmation; no live mutation is wired. Local demo-case creation is clearly labeled and is not a retailer/courier operation.
- `tests/test_conversation_continuity.py` covers multi-turn order/product/return/damage/missing flows, issue ambiguity, invalid inputs, interrupted/resumed requests, per-session isolation, history, and confirmation gating.
- No dependency or PostgreSQL package was added in Phase 4D. Phase 4E.3 is planned to migrate the injected checkpointer; current checkpoints are volatile and do not authenticate session IDs.

## Phase 4D test evidence

Verified with ` .\.venv\Scripts\python.exe -m unittest discover -s tests -v `: **82 tests ran, 81 passed, one optional local-classifier smoke test skipped**. The optional model/Groq test remains skipped by default.


## Phase 4E.1 implementation evidence

- `app/config.py` reads optional `DATABASE_URL` and bounded pool/timeout settings. No database connection is attempted during application import or Flask factory creation.
- `app/db/base.py` provides the declarative ORM base. `app/db/session.py` validates PostgreSQL-only URLs, uses psycopg 3, lazily caches engines/session factories, explicitly manages pool lifecycle, and closes/rolls back sessions safely without implicit commits.
- `alembic.ini`, `migrations/env.py`, the revision template, and `migrations/versions/0001_postgresql_foundation.py` establish migration workflow. The baseline introduces no application tables.
- `DATABASE_URL` is blank in `.env.example`; put local credentials only in ignored `.env` and production credentials in hosted secret configuration.
- Phase 4E.1 does not migrate JSON services, local simulated support records, or LangGraph checkpoints. Phase 4E.2 and 4E.3 remain planned.
- New dependencies: SQLAlchemy, Alembic, and psycopg 3 with its binary distribution. No RAG/checkpointer/database-server package was added.
- Offline PostgreSQL SQL generation via `python -m alembic upgrade head --sql` succeeded using a non-connecting placeholder URL. No live PostgreSQL server was available.

## Phase 4E.1 validation

Full suite ` .\.venv\Scripts\python.exe -m unittest discover -s tests -v ` passed: **90 tests ran, 88 passed, 2 skipped** (optional local classifier and live PostgreSQL connectivity). Database-only suite ` .\.venv\Scripts\python.exe -m unittest tests.test_database tests.test_database_connectivity -v ` passed 7 foundation tests; the live PostgreSQL connectivity check was skipped because `DATABASE_URL` was blank.


## Phase 4E.2 implementation evidence

- `app/db/models.py` defines `products`, `orders`, `support_cases`, `return_request_drafts`, and `refund_records`. Product and order schemas match `data/mock/products.json` and `orders.json`; no customer or order-item model was added because the fixture records provide neither data shape.
- Migration `0002_business_data` follows, and does not replace, `0001_postgresql_foundation`. Offline PostgreSQL SQL generation for `alembic upgrade head --sql` succeeded through both revisions.
- `app/db/repositories.py` provides order/product lookup, parameter-bound product substring search, support-case read/write, return-draft creation, and refund retrieval. Repository sessions are closed by `session_scope`; writes commit explicitly.
- `BUSINESS_DATA_BACKEND=json` remains the default. Setting it to `postgres` explicitly routes order/product/refund reads and simulated support/return record persistence through PostgreSQL; a missing connection does not silently fall back to JSON. Store/policy documents remain JSON and LangGraph checkpoints remain in-memory.
- Current source fixture counts are 3 products, 4 orders, and 3 refunds. The Phase 4E.2 validation at that time had 3 products, 3 orders, and no refunds; no production importer was run in that phase.
- Unit tests use SQLAlchemy metadata checks and a narrow in-memory session double for importer idempotency; they do not emulate PostgreSQL behavior. Existing case-workflow tests continue using their temporary SQLite demo store under the default JSON backend.

## Phase 4E.2 validation

Full suite ` .\.venv\Scripts\python.exe -m unittest discover -s tests -v ` passed: **102 tests ran, 99 passed, 3 skipped** (optional local classifier smoke test, PostgreSQL connectivity, and PostgreSQL repository integration). Offline SQL generation for both Alembic revisions passed. No live PostgreSQL server was configured, so applying the migration and executing repository SQL against PostgreSQL remain unverified.


## Phase 4E.3 implementation evidence

- Added `CHECKPOINT_BACKEND=memory|postgres`; memory remains the default and business-data backend selection remains independent.
- `app/graph/checkpointer.py` uses the compatible `PostgresSaver` with one process-lifetime psycopg 3 `ConnectionPool`, autocommit, dictionary rows, bounded pool/timeouts, and a restricted MessagePack serializer. Initialization failures are explicit and never switch to memory.
- `build_graph(checkpointer=...)` remains injectable. `get_graph()` caches the compiled graph; `close_graph_resources()` clears it and closes checkpoint and SQLAlchemy pools. Cleanup is registered after graph module import and is called explicitly when the local Flask server exits.
- Checkpoint tables and their `checkpoint_migrations` ledger are owned and migrated by `langgraph-checkpoint-postgres` via its repeatable `PostgresSaver.setup()` method, not Alembic. The command `python -m app.graph.checkpointer_cli setup` was run successfully against the configured PostgreSQL database.
- Installed versions: `langgraph==1.2.22`, `langgraph-checkpoint==4.3.2`, `langgraph-checkpoint-postgres==3.2.1`, `psycopg==3.3.9`, `psycopg-pool==3.3.3`. `pip check` reported no broken requirements.
- Unit suite `python -m unittest discover -s tests -q`: **110 tests ran, 108 passed, 2 skipped** (optional classifier smoke test and opt-in process-restart test). PostgreSQL connectivity and business-repository integration tests ran as configured.
- The opt-in `tests.test_postgres_checkpointer_integration` test was run separately against PostgreSQL and **passed**. It started a pending order workflow in one process, closed its graph/pool, resumed it in a new process, and verified another thread did not inherit the pending state. The test removed only the unique thread IDs it created.
- Regression-test isolation: the initial suite attempt ran legacy simulated-workflow tests with the configured PostgreSQL business backend, so some demo support records were written to PostgreSQL and old one-record assertions failed. Those rows were not deleted. The test modules now explicitly select JSON/local demo storage; the subsequent full suite passed without those failures.
- No original classifier/reference projects were changed. No classifier labels, business tables, business data, or existing checkpoint threads were deleted.

## Phase 4E.3 remaining limitations

- `session_id` is still supplied directly as `thread_id`; it is not authentication or authorization. The project has no chat API, user/owner binding, thread authorization, retention/deletion policy, or user-controlled checkpoint deletion.
- The checkpoint pool is process-local and each application worker maintains its own pool, while checkpoint state is shared through PostgreSQL. Pool limits must be sized across all workers for the database's connection budget.
- PostgreSQL checkpoint serialization is restricted to safe built-in MessagePack types; new custom state types may require an explicit safe allowlist or serializer change.
- The database role needs permission to create/update the checkpointer-owned schema on first setup and package migrations. No checkpoint tables are managed by Alembic.


## Phase 5.2 implementation evidence

- Added `app/rag/documents.py` to load only approved UTF-8 Markdown/text files from `knowledge_base/`, chunk them with overlap, and attach stable IDs plus source/topic/content-hash metadata. `knowledge_base/README.md` contains instructions only; no policy or promotion content was invented or indexed.
- Added `app/rag/service.py` for lazy OpenAI embeddings, Pinecone vector upsert/search, score filtering, and DeepSeek generation from JSON-encoded retrieved excerpts. Answers carry source names. Empty/weak retrieval, missing configuration, and provider errors fail to explicit unavailable responses.
- Added `python -m app.rag.ingest`; it reads the configured knowledge directory and upserts batches of 100 into an existing Pinecone index. Empty directories exit without credentials or provider calls. Index creation and automatic deletion/pruning of removed document chunks are not implemented.
- Wired `store_information_node` to RAG without routing changes or classifier label changes. Only explicit development store sales/details queries may display the old fictional fixture when provider configuration is missing; retrieval misses/provider failures never use fixtures, and production never enables this fallback. Order, refund, return, issue, and shipping workflow behavior remains as before.
- Added provider/index/model/chunking settings and blank key placeholders in `.env.example`. Added `openai>=3.49,<4` and `pinecone>=10,<11`; validated installed versions were OpenAI 3.51.0 and Pinecone 10.3.0 and inspected actual method signatures. No real provider calls were made.
- `tests/test_rag.py` uses fake clients. Graph, continuity, and support route tests force a missing-config path so the ordinary suite cannot call paid RAG APIs.
- Full suite `python -m unittest discover -s tests -q`: **127 tests ran, passed; 2 skipped** (optional real-classifier/Groq smoke test and opt-in PostgreSQL checkpoint restart test). `pip check` reported no broken requirements. The empty-corpus indexing command exited successfully without provider credentials. No paid OpenAI, Pinecone, or DeepSeek API calls were made.

## Phase 5.3 analysis evidence

- Inspected the reference controller, retriever, augmenter, generator, tool registry, Flask route, Rasa action, evaluation script, and tests, plus the current RAG service and LangGraph route.
- The reference has bounded retries/hops and optional order/product function calls, but always retrieves first; its confidence is an average vector score, not evidence grading. Source-level defects and integration mismatches are listed in `docs/system_design.md` section 24.
- Documented the proposed minimal LangGraph subgraph: analyze, retrieve, evaluate, one bounded rewrite/retrieval retry, grounded generation or safe unavailable response. Transactional workflows remain selected by the existing classifier and never use RAG for facts.
- Documentation only: no application code, dependencies, classifier labels, reference project files, or provider calls changed. No tests were run because this phase made no runtime changes.


## Phase 5.4 implementation evidence

- Added `app/rag/workflow.py`, a compiled LangGraph subgraph with `analyze_query`, `retrieve_documents`, `evaluate_evidence`, `rewrite_query`, `generate_answer`, and `return_unavailable` nodes. The compiled graph is reused with the existing cached `RAGService`.
- `store_information_node` invokes the subgraph only on the existing `store_information` route. No classifier label or transactional route changed; RAG state does not write pending-workflow fields.
- Reused the existing lazy OpenAI embedding/Pinecone retrieval and DeepSeek client. `RAGService.evaluate_evidence()` asks DeepSeek for strict JSON, validates exact fields/types, and fails closed on malformed output/provider failure. `rewrite_query()` is intent-preserving and is called at most once; retrieval occurs no more than twice.
- Generation uses only retrieved document text and original/current question context; application code attaches source filenames. Insufficient evidence, missing configuration, and provider errors return unavailable responses. Only same-thread relevant prior user text may expand a follow-up query; transactional-looking context is excluded.
- Added offline tests for first-pass success, bounded retry success/exhaustion, empty results, malformed evaluation, evaluator/rewrite provider errors, sources, same-thread context filtering, and preservation of pending workflow state. Existing business and continuity regressions remain covered by the full suite.
- Full suite: `python -m unittest discover -s tests -q` — **138 tests passed; 2 skipped** (opt-in classifier/Groq smoke and PostgreSQL checkpoint restart tests). No paid RAG provider calls were made.
- Remaining: no approved corpus/index or real provider-quality evaluation; evidence judgment is a single DeepSeek evaluator call, not an independently verified faithfulness check. No frontend, API, or deployment changes are included.


## Phase 5.5 implementation evidence

- Split knowledge content into `knowledge_base/approved/` and `knowledge_base/demo/`. The approved directory is empty except for instructions. Added seven clearly labelled fictional demo content Markdown documents across store FAQ, promotions, shipping, returns, refunds and store information, including English and Hindi FAQ examples, plus a demo instructions README.
- Added a small metadata-header parser for `document_id`, `category`, `language`, `last_updated`, and `status`; each chunk also carries filename, relative source, topic, chunk index, content hash, version hash and demo flag. Duplicate document IDs and invalid metadata/date formats are rejected.
- Default application retrieval/indexing uses the approved Pinecone namespace. Demo indexing requires `--demo` in `APP_ENV=development`, writes to a separate namespace, and app retrieval requires explicit `RAG_USE_DEMO_KNOWLEDGE=true` in development. No demo document was sent to an external provider.
- Reindex IDs include a project-owned prefix and content/metadata version. The indexer lists only its prefix in its namespace, skips unchanged IDs, upserts replacements before deleting stale exact IDs, and leaves unrelated IDs untouched. Deleted documents are reconciled when other source files remain; an entirely empty directory requires explicit `--prune-empty`. An upsert failure preserves previous vectors.
- Verified installed Pinecone `Index.list`, `upsert`, `delete`, and `query` method signatures locally. No live index calls were made.
- Added offline cases for English/Hindi query paths, irrelevant and unsupported questions, empty evidence, conflicts, superseded text, prompt injection, bounded retry, source metadata, update/delete sync, empty pruning, failure safety and namespace separation. They validate contracts and graph decisions, not real vector recall or semantic evaluator quality.
- Full suite: `python -m unittest discover -s tests -q` — **153 tests passed; 2 skipped** (opt-in classifier/Groq smoke and PostgreSQL checkpoint restart tests). The offline evaluation summary and limits are in `docs/rag_quality_report.md`.
