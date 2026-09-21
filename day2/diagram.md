# Day 2 diagram

## System architecture

```mermaid
flowchart TD
    User[User / Application]
    Agent[Agent / Orchestrator]

    TaskParser[Task Parser]
    Perception[VLM Perception]
    Grounder[Grounder]
    SkillPlanner[Skill Planner / System2]
    Validator[Plan Validator]
    Runtime[Robot Runtime]
    Router[Recovery Router]

    User -->|"赤い箱を青い箱の隣に置いて"| Agent

    Agent -->|instruction| TaskParser
    TaskParser -->|TaskSpec| Agent

    Agent -->|observe| Perception
    Perception -->|WorldState| Agent

    Agent -->|TaskSpec refs + WorldState| Grounder
    Grounder -->|GroundingResults| Agent

    Agent -->|TaskSpec + WorldState + GroundingResults| SkillPlanner
    SkillPlanner -->|SkillPlan| Agent

    Agent -->|SkillPlan + WorldState| Validator
    Validator -->|valid / invalid| Agent

    Agent -->|SkillRequest: one validated SkillStep + timeout| Runtime
    Runtime -->|SkillResult| Agent

    Agent -->|failed SkillResult| Router
    Router -->|RecoveryAction| Agent

    Agent -.->|REOBSERVE_AND_REGROUND| Perception
    Agent -.->|ASK_SYSTEM2: future replanning path| SkillPlanner
    Agent -.->|ABORT| User
```

- Task Parser = WHAT
- Perception = WHAT EXISTS
- Grounder = WHICH ENTITY
- Skill Planner / System2 = WHICH SKILLS / IN WHAT ORDER
- Plan Validator = CAN THIS PLAN BE EXECUTED AGAINST THE CURRENT WORLD STATE?
- Robot Runtime = EXECUTE ONE SKILL REQUEST
- Recovery Router = WHAT TO DO AFTER FAILURE
- Agent = ORCHESTRATION / CLOSED-LOOP EXECUTION

## Main data contracts

```text
TaskSpec
  -> user's goal; retained across REOBSERVE_AND_REGROUND

WorldState
  -> current observed world; discarded and rebuilt after re-observation

GroundingResults
  -> one result per requested ref
  -> RESOLVED / NOT_FOUND / AMBIGUOUS

SkillPlan
  -> ordered SkillSteps produced by System2
  -> e.g. PICK(obj_0) -> PLACE(obj_0, NEXT_TO, obj_2)

SkillRequest
  -> one SkillStep + runtime metadata (currently timeout_sec)
  -> Agent dispatches requests one step at a time

SkillResult
  -> SUCCEEDED / PRECONDITION_FAILED / FAILED / TIMEOUT / CANCELLED
  -> includes a reason used by Recovery Router

RecoveryAction
  -> REOBSERVE_AND_REGROUND / ASK_SYSTEM2 / ABORT
```

## Recovery loop implemented in Day 2

```mermaid
flowchart TD
    Execute[Execute SkillRequest]
    Result{SkillResult}
    Router[Recovery Router]
    Observe[Perception / Re-observe]
    Ground[Ground again]
    Plan[Re-plan]
    Validate[Validate new SkillPlan]
    Abort[Abort / return failure]
    Success[Continue / task success]

    Execute --> Result
    Result -->|SUCCEEDED| Success
    Result -->|failure| Router
    Router -->|REOBSERVE_AND_REGROUND| Observe
    Observe -->|new WorldState| Ground
    Ground -->|new GroundingResults| Plan
    Plan -->|new SkillPlan| Validate
    Validate -->|valid| Execute
    Router -->|ASK_SYSTEM2| Abort
    Router -->|ABORT| Abort
```

`REOBSERVE_AND_REGROUND` is implemented as a bounded retry loop (`max_retries`).
`ASK_SYSTEM2` is routed but full failure-aware System2 replanning is deferred.

## Timing boundary / RT mini-lab

```mermaid
flowchart TD
    System2[System2 / LLM\nslow + nondeterministic]
    Agent2[Agent / Orchestrator]
    Runtime2[Robot Runtime\ntime-constrained execution]
    Controller[Controller / Safety\nfaster loop]
    Robot[Robot]

    System2 -->|SkillPlan| Agent2
    Agent2 -->|SkillRequest| Runtime2
    Runtime2 -->|command / setpoint| Controller
    Controller -->|actuation| Robot
    Robot -->|state / sensors| Controller
```

Day 2 RT mini-lab used a 100 Hz (10 ms) best-effort Python loop with absolute scheduling.
Measured quantities were loop period, execution time, scheduling jitter, and deadline misses.
Under CPU load, the long-term mean period stayed near 10 ms while individual periods and maximum jitter increased. This demonstrates that an average 100 Hz rate is not a real-time guarantee.

The architectural consequence is that slow, variable-latency System2/LLM work must not sit inside a time-constrained robot/control loop. The boundary is expressed through coarse-grained `SkillPlan` / `SkillRequest` interfaces instead.
