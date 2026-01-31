# Pidcast CLI Reimagined: An Ultrathink Analysis

## The Core Problem

You're forgetting flags. This isn't a you problem - it's a design problem. Let's examine why:

### Current Pain Points

1. **Flag Names Are Implementation-Driven, Not User-Driven**
   - `--save_to_obsidian` vs `--save` - Why two different save flags?
   - `--analysis_type` requires memorizing values from YAML (`executive_summary`)
   - `--prompts_file` instead of just `--prompts`
   - `--output_dir` vs `--save_to_obsidian` creates mental model conflict

2. **Cognitive Overload on Common Operations**
   ```bash
   # What you actually want to say:
   "Process this podcast and save it"

   # What you have to type:
   pidcast "URL" --save_to_obsidian --analysis_type executive_summary --verbose
   ```

3. **Hidden Complexity in Subcommands**
   ```bash
   # Library workflow requires remembering:
   pidcast lib add "URL"           # Different from main workflow
   pidcast lib sync --show ID      # "show" as a flag value? or "show" as subcommand?
   pidcast lib show ID             # Wait, this is different "show"
   ```

4. **No Progressive Disclosure**
   - Beginners need simplicity: `pidcast URL`
   - Power users want control: all the flags
   - Current design treats everyone the same

5. **Flag Value Inconsistency**
   - `--analysis_type` takes string values that must match YAML keys
   - No autocomplete hints
   - No validation until runtime
   - No listing of available types

## The Vision: A CLI That Thinks Like You Do

### Design Principle 1: Convention Over Configuration

**Bad (Current):**
```bash
pidcast "URL" --save_to_obsidian --analysis_type executive_summary
```

**Good (Proposed):**
```bash
pidcast "URL"  # Auto-saves to Obsidian if configured, uses default analysis
```

The tool should have smart defaults based on:
- Environment variables (already there)
- Config file (`~/.config/pidcast/config.yaml`)
- Contextual intelligence (if Obsidian path is set, assume you want to save there)

### Design Principle 2: Natural Language Flags

**Bad (Current):**
```bash
--save_to_obsidian
--analysis_type executive_summary
--skip_analysis_on_error
```

**Good (Proposed):**
```bash
-o, --obsidian           # Short and memorable
-a, --analysis TYPE      # Clear what comes after
-c, --continue-on-error  # Matches git's style
```

### Design Principle 3: Interactive When Ambiguous

**Current:** You must know all flag names and values upfront.

**Proposed:** If you don't specify, the tool asks:

```bash
$ pidcast "URL"
✓ Downloaded: The Future of AI (1:23:45)
✓ Transcribed: 45,231 words

? Analysis type: (Use arrow keys)
❯ Executive Summary (Quick overview)
  Detailed Notes (Full breakdown)
  Key Quotes (Memorable moments)
  Skip analysis

? Save to: (Use arrow keys)
❯ Obsidian Vault
  Local directory (./data/transcripts)
  Both
```

### Design Principle 4: Composable Commands

Instead of massive flag lists, use Unix philosophy:

```bash
# Current (complex)
pidcast lib sync --show 5 --dry-run --force --backfill 10 --output_dir ./out --verbose

# Proposed (composable)
pidcast sync 5 --preview          # Dry run with clear name
pidcast sync 5 --force --limit 10 # Backfill -> limit (clearer)
pidcast sync 5 > ./out            # Unix redirect for output
```

## The Solution: Three-Tier CLI Design

### Tier 1: Simple Mode (Zero Flags Required)

```bash
# Just works with smart defaults
pidcast "https://youtube.com/watch?v=..."

# Interactive prompts fill in the gaps
# Remembers your choices in ~/.config/pidcast/config.yaml
```

### Tier 2: Power User Shortcuts

```bash
# Short flags for common operations
pidcast "URL" -o              # Save to Obsidian
pidcast "URL" -a detailed     # Use detailed analysis
pidcast "URL" -f              # Force (already exists!)
pidcast "URL" -v              # Verbose (already exists!)

# Chain them
pidcast "URL" -ofv
```

### Tier 3: Full Control

```bash
# Long flags for scripts and precise control
pidcast "URL" \
  --analysis executive_summary \
  --output ~/custom/path \
  --groq-model llama-3.3-70b-versatile \
  --no-save
```

## Specific Improvements

### 1. Fix the Save Flags Mess

**Current Confusion:**
- `--save` → "Save analysis output to file"
- `--save_to_obsidian` → "Save to Obsidian vault"
- Default behavior → Save transcript to output_dir

**Proposed Clarity:**
```bash
# Destination flags (mutually exclusive)
-o, --obsidian              # Save to Obsidian (analysis + transcript)
-d, --dir PATH              # Save to directory (default: ./data/transcripts)

# What to save
--analysis-only             # Skip transcript file
--transcript-only           # Skip analysis (same as --no-analyze)

# Examples
pidcast "URL"               # Save both to default dir
pidcast "URL" -o            # Save both to Obsidian
pidcast "URL" --analysis-only -o  # Only analysis to Obsidian
```

### 2. Make Analysis Types Discoverable

**Current Problem:**
- Must know YAML keys: `executive_summary`, etc.
- No way to list available types

**Proposed Solution:**

```bash
# List available analysis types
pidcast --list-analyses
# or
pidcast -L

Output:
Available analysis types:
  executive    Executive Summary (default)
  detailed     Detailed breakdown with timestamps
  quotes       Key quotes and memorable moments

Use: pidcast URL --analysis TYPE
```

**Autocomplete Support:**
```bash
# Add shell completion script
pidcast --install-completion bash
# Now:
pidcast URL --analysis <TAB>
  → executive  detailed  quotes
```

### 3. Simplify Library Commands

**Current Confusion:**
```bash
pidcast lib add "URL"
pidcast lib show ID
pidcast lib sync --show ID
```

Notice: "show" is both a noun (subcommand) and a value (flag). Confusing!

**Proposed Hierarchy:**

```bash
# Library is a first-class workflow
pidcast add "RSS_URL"              # Simpler
pidcast list                       # No "lib" prefix needed
pidcast info ID                    # "show" → "info" (clearer)
pidcast sync [ID]                  # Optional positional arg
pidcast get "Show Name" --latest   # Get episode from library

# The "lib" prefix becomes optional
pidcast lib add "URL"              # Still works for backwards compat
```

### 4. Add Aliases for Common Workflows

```bash
# Transcribe and save to Obsidian (most common operation)
pidcast quick "URL"
# Equivalent to: pidcast "URL" -o -a executive

# Process latest from library
pidcast latest "Lex Fridman"
# Equivalent to: pidcast get "Lex Fridman" --latest -o

# Analyze existing transcript
pidcast analyze transcript.md
# Equivalent to: pidcast --analyze_existing transcript.md
```

### 5. Smart Model Selection

**Current Problem:**
```bash
--groq_model llama-3.3-70b-versatile  # Must memorize full name
```

**Proposed Solution:**

```bash
# Use fuzzy matching
-m, --model llama33    # Matches llama-3.3-70b-versatile
-m llama8              # Matches llama-3.1-8b-instant
-m gpt120              # Matches openai/gpt-oss-120b

# List models
pidcast --list-models
pidcast -M

Output:
Available models (quality order):
  gpt120       GPT-OSS 120B (best quality, 8k TPM)
  compound     Groq Compound (high quality, 70k TPM)
  gpt20        GPT-OSS 20B (good quality, 8k TPM)
  llama33      Llama 3.3 70B (good quality, 12k TPM)
  llama8       Llama 3.1 8B (fast, 6k TPM)
```

### 6. Better Sync Workflow

**Current:**
```bash
pidcast lib sync --show ID --dry-run --force --backfill N
```

**Proposed:**
```bash
# Clearer flag names
pidcast sync [ID]          # Sync all or specific show
  --preview                # Dry run
  --reprocess              # Force reprocessing
  --limit N                # Max episodes to process
  --no-digest              # Skip digest generation (keep)

# Examples
pidcast sync --preview     # See what would sync
pidcast sync 5 --limit 10  # Sync show 5, max 10 episodes
pidcast sync --reprocess   # Reprocess all
```

## Implementation Strategy

### Phase 1: Non-Breaking Enhancements (Week 1)

1. **Add short flags** (keep long ones for compatibility)
   - `-o` alias for `--save_to_obsidian`
   - `-a` alias for `--analysis_type`
   - `-m` alias for `--groq_model`

2. **Add listing commands**
   - `--list-analyses` / `-L`
   - `--list-models` / `-M`

3. **Add fuzzy matching for model names**
   - Backend: fuzzy matcher for model strings
   - Fallback to exact match if no fuzzy match

### Phase 2: Smart Defaults (Week 2)

1. **Config file support**
   - `~/.config/pidcast/config.yaml`
   - Store: default analysis type, prefer Obsidian, default model

2. **Environment-based defaults**
   - If `OBSIDIAN_VAULT_PATH` set → default to saving there
   - Smart choice: "analysis + transcript to Obsidian" or "just analysis"

3. **Contextual defaults**
   - `pidcast URL` → uses config defaults
   - `pidcast analyze FILE` → auto-detect from flags

### Phase 3: Interactive Mode (Week 3)

1. **Optional prompts** when flags missing
   - Only in interactive TTY
   - Scriptable mode: fail fast

2. **Remember choices**
   - "Remember this choice?" prompt
   - Stores in config file

### Phase 4: Command Restructure (Week 4)

1. **Flatten common commands**
   - `pidcast add` instead of `pidcast lib add`
   - Keep `lib` prefix as alias

2. **Add workflow aliases**
   - `pidcast quick URL`
   - `pidcast latest "Show"`
   - `pidcast analyze FILE`

### Phase 5: Polish (Week 5)

1. **Shell completion**
   - Bash/Zsh completion scripts
   - Fish support

2. **Better error messages**
   - Suggest correct flag if typo detected
   - "Did you mean: --analysis executive?"

3. **Help redesign**
   - `pidcast help` → friendly guide
   - `pidcast help workflows` → common patterns
   - `pidcast --help` → traditional argparse output

## The Result

### Before (Current):
```bash
# Cognitive load: HIGH
# Must remember: flag names, flag values, subcommand structure
pidcast "URL" --save_to_obsidian --analysis_type executive_summary --verbose
pidcast lib sync --show 5 --backfill 10
pidcast --analyze_existing transcript.md --analysis_type detailed
```

### After (Proposed):
```bash
# Cognitive load: LOW
# Memorable short flags, smart defaults, fuzzy matching
pidcast "URL"                    # Just works
pidcast "URL" -o -a exec -v      # Short flags, fuzzy values
pidcast sync 5 --limit 10        # Clearer language
pidcast analyze transcript.md    # Natural command
```

## Validation Metrics

How do we know this is better?

1. **Time to First Success**
   - New user → successful transcription
   - Target: < 2 minutes (including setup)

2. **Command Recall Rate**
   - Can user reproduce command without looking?
   - Target: > 80% after 3 uses

3. **Flag Autocomplete Usage**
   - % of users who install completions
   - Target: > 50%

4. **Error Rate**
   - Typos, wrong flags, wrong values
   - Target: < 10% of commands

5. **Help Usage**
   - How often users run `--help`
   - Target: Decreasing trend over time

## The Philosophy

Good CLI design is invisible. Users shouldn't think about the tool - they should think about their task.

**Current state:** "How do I tell pidcast to save to Obsidian?"
**Desired state:** "I need to transcribe this podcast" → `pidcast URL` → Done.

The best CLIs are:
1. **Intuitive** - Obvious what to do next
2. **Forgiving** - Typos corrected, omissions filled in
3. **Fast** - Minimal typing for common tasks
4. **Powerful** - Full control when needed
5. **Teachable** - Each use teaches the next

---

## Immediate Next Steps (This Week)

If you want to start improving TODAY:

1. **Quick Win: Short Flags** (30 minutes)
   ```python
   parser.add_argument("-o", "--save_to_obsidian", ...)
   parser.add_argument("-a", "--analysis", dest="analysis_type", ...)
   parser.add_argument("-m", "--model", dest="groq_model", ...)
   ```

2. **Quick Win: List Commands** (1 hour)
   ```python
   parser.add_argument("-L", "--list-analyses", action="store_true")
   parser.add_argument("-M", "--list-models", action="store_true")
   ```

3. **Quick Win: Better Help** (30 minutes)
   - Update epilog with common examples using short flags
   - Add "Common Workflows" section

These three changes alone would reduce your cognitive load by ~50%.

Want me to implement these?
