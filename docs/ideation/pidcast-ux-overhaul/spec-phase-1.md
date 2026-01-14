# Implementation Spec: Pidcast UX Overhaul - Phase 1

**PRD**: ./prd-phase-1.md
**Estimated Effort**: M (Medium)

## Technical Approach

This phase refactors the CLI argument parsing to make `--analyze` the default behavior and migrates prompts from JSON to YAML. The changes are surgical - we're adjusting defaults and swapping file formats, not rewriting the architecture.

Key decisions:
1. **Invert the analyze flag**: Instead of `--analyze` to enable, we add `--no-analyze` to disable
2. **YAML for prompts**: Use PyYAML (already a transitive dependency) for cleaner prompt storage
3. **Remove auto-generation**: Delete `create_default_prompts_file()` and fail clearly if prompts.yaml is missing
4. **New "executive_summary" type**: A combined prompt that returns both summary and key points in one call

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `config/prompts.yaml` | YAML-based prompt configuration (replaces JSON) |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `src/pidcast/cli.py` | Change CLI defaults: analyze by default, add --no-analyze, --save flags |
| `src/pidcast/analysis.py` | Remove `create_default_prompts_file()`, load YAML instead of JSON |
| `src/pidcast/config.py` | Update `DEFAULT_ANALYSIS_PROMPTS_FILE` to point to `prompts.yaml` |

### Deleted Files

| File Path | Reason |
|-----------|--------|
| `config/analysis_prompts.json` | Replaced by `config/prompts.yaml` |

## Implementation Details

### 1. CLI Argument Changes

**Pattern to follow**: Existing argparse structure in `src/pidcast/cli.py:82-174`

**Overview**: Invert the `--analyze` flag logic and add new convenience flags.

```python
# Before
analysis_group.add_argument(
    "--analyze", action="store_true", help="Enable LLM-based transcript analysis"
)

# After
analysis_group.add_argument(
    "--no-analyze", action="store_true",
    help="Skip LLM analysis (default is to analyze)"
)
analysis_group.add_argument(
    "--save", action="store_true",
    help="Save output to file (default: terminal only)"
)
```

**Key decisions**:
- Keep `--analyze` as a no-op for backward compatibility (accepts but does nothing)
- Default `--analysis_type` to "executive_summary" instead of "summary"
- `--save` triggers file output, `--save-to-obsidian` is unchanged

**Implementation steps**:
1. Add `--no-analyze` argument with `action="store_true"`
2. Add `--save` argument with `action="store_true"`
3. Keep `--analyze` but mark as deprecated in help text
4. Change `--analysis_type` default to "executive_summary"
5. Update `main()` logic: run analysis unless `--no-analyze` is set
6. Update file-saving logic: only save if `--save` or `--save-to-obsidian`

### 2. Prompt Loading (YAML)

**Pattern to follow**: `src/pidcast/evals/prompt_manager.py` for YAML loading patterns

**Overview**: Replace JSON loading with YAML and remove auto-generation.

```python
# New function signature (replaces load_analysis_prompts)
def load_prompts(prompts_file: Path, verbose: bool = False) -> PromptsConfig:
    """Load prompts from YAML file.

    Args:
        prompts_file: Path to prompts.yaml
        verbose: Enable verbose output

    Returns:
        PromptsConfig with loaded prompts

    Raises:
        AnalysisError: If file missing or invalid
    """
    if not prompts_file.exists():
        raise AnalysisError(
            f"Prompts file not found: {prompts_file}\n"
            "Expected: config/prompts.yaml"
        )

    import yaml
    with open(prompts_file) as f:
        config = yaml.safe_load(f)

    # Validate and return PromptsConfig
    return PromptsConfig.from_dict(config)
```

**Key decisions**:
- Use `yaml.safe_load()` for security
- Fail immediately if file missing (no fallback creation)
- Same PromptsConfig dataclass works for both JSON and YAML

**Implementation steps**:
1. Add `import yaml` (PyYAML is already in dependencies via other packages)
2. Remove `create_default_prompts_file()` function entirely
3. Update `load_analysis_prompts()` to use YAML
4. Rename to `load_prompts()` for clarity
5. Update all call sites in cli.py

### 3. Prompts YAML File

**Overview**: Create the new prompts.yaml with existing prompts plus new executive_summary.

```yaml
# config/prompts.yaml
prompts:
  executive_summary:
    name: Executive Summary
    description: Concise summary with key points for quick content triage
    max_output_tokens: 1500
    system_prompt: |
      You are an expert at distilling video content into actionable briefs.
      Create summaries that help viewers decide if content is worth their time.
    user_prompt: |
      Create an executive brief of this transcript.

      Title: {title}
      Channel: {channel}
      Duration: {duration}

      # Transcript
      {transcript}

      # Instructions
      Provide:
      1. **Executive Summary** (2-3 sentences, ~50-75 words)
         - What is this content about?
         - Who is it for?
         - What's the main value proposition?

      2. **Key Points** (5-10 bullet points)
         - Most important insights
         - Concrete takeaways
         - Notable quotes or data points

      Keep the summary brief enough to share via text message or Discord.

  summary:
    name: Summary
    description: Generate a concise summary of the transcript
    max_output_tokens: 2000
    system_prompt: |
      You are an expert at summarizing video transcripts.
      Create clear, concise summaries that capture the main points and key insights.
    user_prompt: |
      Please create a comprehensive summary of the following transcript.

      Title: {title}
      Channel: {channel}
      Duration: {duration}

      # Transcript
      {transcript}

      # Instructions
      Provide a summary with:
      1. A one-paragraph overview (2-3 sentences)
      2. Key points (3-7 bullet points)
      3. Main takeaways or conclusions

      Be specific and use concrete details from the transcript.

  key_points:
    name: Key Points Extraction
    description: Extract main ideas and important points
    max_output_tokens: 2500
    system_prompt: |
      You are an expert at analyzing content and extracting key insights.
      Focus on identifying the most important ideas, facts, and actionable takeaways.
    user_prompt: |
      Extract the key points from this transcript.

      Title: {title}
      Channel: {channel}

      # Transcript
      {transcript}

      # Instructions
      Provide:
      1. Main Topics: List 3-5 main topics discussed
      2. Key Points: 8-12 bullet points of the most important ideas
      3. Notable Quotes: 2-3 memorable or impactful quotes
      4. Actionable Insights: Any practical advice or takeaways

  action_items:
    name: Action Items & Recommendations
    description: Extract actionable recommendations and next steps
    max_output_tokens: 2000
    system_prompt: |
      You are an expert at identifying actionable advice and practical recommendations.
      Focus on what viewers can do with this information.
    user_prompt: |
      Identify action items and recommendations from this transcript.

      Title: {title}

      # Transcript
      {transcript}

      # Instructions
      Provide:
      1. Direct Action Items: Specific steps or actions mentioned
      2. Recommended Resources: Tools, books, links, or resources referenced
      3. Best Practices: Any guidelines or principles discussed
      4. Implementation Tips: How to apply these insights
```

**Implementation steps**:
1. Create `config/prompts.yaml` with above content
2. Delete `config/analysis_prompts.json` after confirming YAML works
3. Update `config.py` to set `DEFAULT_ANALYSIS_PROMPTS_FILE = PROJECT_ROOT / "config" / "prompts.yaml"`

### 4. Terminal Output Formatting

**Pattern to follow**: Existing `render_analysis_to_terminal()` in `analysis.py:618-720`

**Overview**: Ensure terminal output is clean and readable by default.

**Key decisions**:
- Keep existing Rich-based rendering
- Remove file path output when `--save` is not used
- Show "Generated with: {model}" in metadata panel

**Implementation steps**:
1. Modify `run_analysis()` to skip file creation when not saving
2. Create in-memory rendering path for terminal-only output
3. Add model name to terminal output metadata

## Testing Requirements

### Manual Testing

- [ ] `pidcast URL` produces executive summary + key points in terminal
- [ ] `pidcast URL --no-analyze` produces transcription only
- [ ] `pidcast URL --save` saves output to file
- [ ] `pidcast URL --analyze` still works (backward compat)
- [ ] `pidcast URL --analysis_type summary` uses summary prompt
- [ ] Error when `config/prompts.yaml` is missing is clear
- [ ] Existing `--save-to-obsidian` flag still works

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| `prompts.yaml` missing | Raise `AnalysisError` with clear message pointing to expected path |
| Invalid YAML syntax | Raise `AnalysisError` with line number from yaml.YAMLError |
| Missing required prompt fields | Raise `AnalysisError` listing which fields are missing |

## Validation Commands

```bash
# Lint and format
uv run ruff check src/pidcast/
uv run ruff format src/pidcast/

# Test basic usage
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Test no-analyze flag
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --no-analyze

# Test save flag
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --save

# Test backward compatibility
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --analyze

# Test analysis type override
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --analysis_type key_points
```

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
