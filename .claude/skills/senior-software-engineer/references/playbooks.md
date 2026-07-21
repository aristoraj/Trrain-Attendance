# Engineering Mode Playbooks

Detailed checklists for each mode. Use these when the task is substantial; for small tasks the condensed table in SKILL.md is enough. In every mode, Core Principles still apply — especially Simplicity First (production-grade ≠ gold-plated).

## Build Mode (full-stack feature / MVP)

Design first, then build minimal.

1. **Architecture pass (before code):**
   - System architecture (components and how they talk)
   - File/folder structure
   - Database schema
   - API endpoints (routes, methods, payloads)
   - UI architecture (if applicable)
2. **Build pass:** the most minimal implementation that could realistically evolve. Minimal ≠ sloppy: proper error handling for real scenarios, sensible naming, no dead code.
3. **Deliver:** working code + brief note on what was deliberately left out and why.

Anti-pattern: building "for millions of users" on day one. Build so scaling later doesn't require a rewrite — that's it.

## Debug Mode (production-level debugging)

Handle it like a critical incident: methodical, no guessing.

1. **Understand** — what does the code actually do (not what it's supposed to do)?
2. **Reproduce** — write a failing test or minimal repro if possible.
3. **Trace the real root cause** — follow data flow to origin; don't stop at the first suspicious line. Distinguish symptom from cause.
4. **Explain why the failure happens** — if you can't explain it, you haven't found it.
5. **Check hidden edge cases** — does the root cause break anything else? Are there sibling bugs?
6. **Fix robustly and minimally** — fix the cause, not the symptom. Surgical change.
7. **Verify** — failing test now passes; existing tests still pass.

Deliver: root cause analysis (2–4 sentences) + fix + how it was verified.

## Review / Audit Mode (codebase audit)

Approach as a senior engineer who just joined an unfamiliar codebase.

1. **Reverse-engineer first:** map the architecture and complete data flow before judging anything.
2. **Identify:**
   - Bad architecture decisions (and why they're bad *here*)
   - Duplicate logic
   - Performance bottlenecks
   - Scalability risks
   - Maintainability issues
   - Security red flags (hardcoded credentials, unvalidated input, exposed PII)
3. **Deliver:** architecture breakdown → critical problem areas (prioritized: fix-now / fix-soon / nice-to-have) → concrete refactoring strategies.
4. **Do not change functionality** when providing improved code — quality, scalability, maintainability only.

## Refactor Mode (messy code → clean architecture)

Mission: separate concerns, increase modularity, reduce tight coupling, improve long-term maintainability.

Hard rules:
- **Behavior must not change.** Tests pass before and after (write characterization tests first if none exist).
- Refactor in reviewable steps, not one giant diff.
- Match the language/framework's idioms, not a textbook pattern imported wholesale.

Deliver: new structure (folder layout if relevant) + refactored code + short explanation of each architectural improvement and what it buys.

## Optimize Mode (performance engineering)

Goals: speed, memory, scalability, rendering, execution efficiency — in the places that matter.

1. **Measure or reason first** — identify actual bottlenecks: inefficient logic (O(n²) where O(n) works), unnecessary re-rendering, expensive operations in hot paths, N+1 queries, memory leaks.
2. **Rank by impact.** Fix the top items; ignore micro-optimizations that complicate code.
3. **Preserve behavior and readability** — an unreadable "fast" version is a net loss unless profiled and necessary.

Deliver: bottleneck breakdown → optimization applied per item → expected impact → scalability recommendations for later.

## Architect Mode (systems design)

For backends and system infrastructure:

- System architecture and component structure
- Data flow between components
- API design (contracts, versioning, error shapes)
- Database schema (indexes, relationships, growth pattern)
- Caching strategy (what, where, invalidation)
- Failure modes: what happens when each dependency is down?

Then: build or specify the **minimal implementation that could realistically scale** — name the future scaling path (e.g., "swap SQLite→Postgres here") instead of pre-building it.

## Frontend Mode (production UI)

Every component/interface handles:

- **Loading states** (skeleton/spinner, no layout jump)
- **Empty states** (helpful, not blank)
- **Error states** (actionable message, retry path)
- **Edge cases** (long text, zero items, thousands of items, slow network)
- **Responsive design** (mobile-first where relevant)
- **Accessibility** (semantic HTML, keyboard nav, labels, contrast)
- **Reusability** (clean props/API design, no hidden coupling)

Deliver: component architecture → props/API design → implementation → usage example.

## Security Mode (production security audit)

Inspect for:

- **Injection** (SQL/NoSQL/command/XSS)
- **Authentication & authorization flaws** (broken access control, missing checks per-route, insecure sessions/tokens)
- **API weaknesses** (mass assignment, missing rate limits, verbose errors leaking internals)
- **Sensitive data exposure** (secrets/keys in code, PII in logs, unencrypted storage, plaintext passwords)
- **Infrastructure risks** (permissive CORS, missing security headers, outdated dependencies)

Deliver a vulnerability report: finding → severity (Critical/High/Medium/Low) → realistic attack scenario → secure fix (code). Run this mode unprompted whenever the task touches auth, payments, uploads, or personal data.

## Deploy Mode (DevOps / production readiness)

- Deployment architecture (environments, secrets management)
- CI/CD pipeline (build → test → deploy, gated)
- Monitoring & logging (what signals indicate trouble; structured logs)
- Reliability (health checks, graceful degradation, retry policies)
- Rollback plan (how to undo a bad deploy in minutes)
- Containerization (Docker/K8s) only if the project's scale justifies it

Deliver: production deployment checklist tailored to the actual stack — not a generic Kubernetes sermon for a single-server app.

## Tech Lead Mode (before code exists)

For ambiguous, large, or high-stakes requests. Think like the person maintaining this for 5+ years.

Before any code:
1. Ask clarifying questions (batch them, don't drip).
2. Challenge decisions that look wrong — with reasoning, not attitude.
3. Identify scaling and maintenance risks.
4. Present 2–3 approaches with tradeoff analysis (cost, complexity, time, risk).
5. Recommend one, and say why.
6. Prioritize simplicity in the recommendation.

Then deliver: decision → tradeoffs considered → recommended architecture → implementation plan → build.

## Multi-Perspective Pass (for complex builds)

For substantial implementations, run four internal passes before delivering — as one coherent flow, not four reports:

1. **Architect:** is the design sound and scalable-later?
2. **Engineer:** build it.
3. **Reviewer:** senior-level critique of your own output — bugs, edge cases, unclear naming, principle violations.
4. **Optimizer:** any real bottlenecks or waste? Fix, then deliver.

Deliver only the final result + brief notes on what the review pass caught and changed.
