# PRD: Pidcast UX Overhaul - Phase 1

**Contract**: ./contract.md
**Phase**: 1 of 3
**Focus**: Simplified CLI defaults and prompt management cleanup

## Phase Overview

This phase tackles the foundational UX improvements that make the most common workflow frictionless. After this phase, users can run `pidcast URL` and immediately get useful output in their terminal without remembering flags.

We're also cleaning up prompt management by migrating from JSON to YAML and removing the confusing auto-generation logic. This is sequenced first because subsequent phases (chunking, model fallback) will add new prompts, and we want the cleaner system in place first.

The phase delivers immediate value: the primary "quick triage" workflow becomes a single command.

## User Stories

1. As a YouTube browser, I want to run `pidcast URL` with no flags so that I can quickly decide if a video is worth watching
2. As a user, I want to see executive summary + key points by default so that I get enough context to make a watch decision
3. As a developer, I want prompts in a readable YAML file so that I can easily modify and understand them

## Functional Requirements

### CLI Defaults

- **FR-1.1**: `pidcast URL` without flags shall analyze the video and output to terminal (no file save)
- **FR-1.2**: Default analysis type shall be "executive_summary" which includes a concise summary + key points
- **FR-1.3**: `--no-analyze` flag shall skip LLM analysis (for users who only want transcription)
- **FR-1.4**: `--save` flag shall save output to file (previously default behavior)
- **FR-1.5**: `--save-obsidian` shall save to Obsidian vault (existing behavior, now explicit)
- **FR-1.6**: Existing flags shall continue to work for backward compatibility

### Prompt Management

- **FR-1.7**: All prompts shall be stored in `config/prompts.yaml`
- **FR-1.8**: Remove `analysis_prompts.yaml` auto-generation logic from codebase
- **FR-1.9**: Add new "executive_summary" prompt type that combines summary + key points
- **FR-1.10**: Prompt loading shall fail clearly if YAML file is missing (no silent fallback)

### Output Format

- **FR-1.11**: Terminal output shall be formatted for readability (headers, spacing, clear sections)
- **FR-1.12**: Executive summary shall be concise enough to share via text/Discord (~200-400 words)
- **FR-1.13**: Key points shall be bullet points (5-10 items)

## Non-Functional Requirements

- **NFR-1.1**: CLI response time for `--help` shall be under 100ms
- **NFR-1.2**: Prompt YAML file shall be human-readable without documentation
- **NFR-1.3**: Error messages shall clearly indicate what went wrong and how to fix it

## Dependencies

### Prerequisites

- None (first phase)

### Outputs for Next Phase

- Clean prompt management system in YAML
- New CLI flag structure
- Executive summary prompt template

## Acceptance Criteria

- [ ] `pidcast URL` produces executive summary + key points in terminal
- [ ] `pidcast URL --no-analyze` produces transcription only
- [ ] `pidcast URL --save` saves output to file
- [ ] `config/prompts.yaml` contains all prompts
- [ ] No auto-generation code remains in codebase
- [ ] Existing `--analyze` flag still works (now effectively a no-op since it's default)
- [ ] `--analysis_type` flag still works for other analysis types

---

*Review this PRD and provide feedback before spec generation.*
