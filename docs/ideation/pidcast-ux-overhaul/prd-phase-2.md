# PRD: Pidcast UX Overhaul - Phase 2

**Contract**: ./contract.md
**Phase**: 2 of 3
**Focus**: Smart model fallback chain with quality priority

## Phase Overview

This phase implements reliable model selection that handles rate limits gracefully. Currently, when the preferred model hits rate limits, the behavior is inconsistent and often fails. After this phase, the system automatically falls back through a quality-prioritized chain of models.

This is sequenced after Phase 1 because it builds on the clean prompt system and needs the new CLI structure in place. It's before Phase 3 (chunking) because chunking will make many more API calls, amplifying the need for robust fallback.

Value delivered: Users no longer see rate limit failures—the system finds a working model automatically.

## User Stories

1. As a user, I want the tool to automatically try another model when rate-limited so that I don't have to manually retry
2. As a user, I want to see which model was used so that I understand the quality level of my results
3. As a user, I want the fallback to prioritize quality so that I get the best available results

## Functional Requirements

### Model Fallback Chain

- **FR-2.1**: Default model chain shall be: gptoss120b → compound → llama-3.3-70b → llama-3.1-8b
- **FR-2.2**: On rate limit error, system shall wait briefly then try next model in chain
- **FR-2.3**: On successful fallback, system shall log which model was used
- **FR-2.4**: If all models fail, system shall provide clear error with retry guidance

### Rate Limit Handling

- **FR-2.5**: Rate limit detection shall recognize both 429 status and TPM exceeded errors
- **FR-2.6**: Before calling API, estimate tokens and check against known TPM limits
- **FR-2.7**: If estimated tokens exceed model's TPM, skip that model (don't waste a call)
- **FR-2.8**: Implement exponential backoff between retries on same model

### Configuration

- **FR-2.9**: Model chain shall be configurable via `config/models.yaml`
- **FR-2.10**: Each model entry shall include: name, TPM limit, priority order
- **FR-2.11**: `--model` flag shall override chain and use only specified model

### Logging

- **FR-2.12**: Log model selection decisions at INFO level
- **FR-2.13**: Log rate limit encounters at WARNING level
- **FR-2.14**: Final output shall indicate which model produced the result

## Non-Functional Requirements

- **NFR-2.1**: Model fallback shall add no more than 5 seconds overhead per retry
- **NFR-2.2**: TPM estimation shall be accurate within 20% of actual usage
- **NFR-2.3**: Fallback logic shall be testable without making actual API calls

## Dependencies

### Prerequisites

- Phase 1 complete (clean prompt system, new CLI structure)

### Outputs for Next Phase

- Robust model selection that handles rate limits
- Token estimation utilities
- Model configuration system

## Acceptance Criteria

- [ ] Rate limit on gptoss120b automatically triggers compound fallback
- [ ] All 4 models in chain are tried before final failure
- [ ] TPM pre-check skips models that would exceed limits
- [ ] `--model llama-3.3-70b` uses only that model (no fallback)
- [ ] Output shows "Generated with: {model_name}"
- [ ] `config/models.yaml` exists with all model configurations
- [ ] Rate limit handling works for both 429 and TPM errors

---

*Review this PRD and provide feedback before spec generation.*
