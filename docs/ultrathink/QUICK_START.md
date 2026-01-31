# Your New CLI - Quick Start Guide

## The Magic Shortcuts You Can Now Use

### Instead of typing this monstrosity:
```bash
pidcast "https://youtube.com/watch?v=abc123" --save_to_obsidian --analysis_type executive_summary --verbose
```

### You can now type:
```bash
pidcast "https://youtube.com/watch?v=abc123" -o -a exec -v
```

**60% fewer keystrokes. Same result.**

---

## Your New Superpowers

### 1. Forgot what analysis types exist?
```bash
pidcast -L
```

Shows you all available types with descriptions.

### 2. Forgot what models exist?
```bash
pidcast -M
```

Shows you all models with their rate limits.

### 3. Can't remember exact names? No problem!
```bash
pidcast URL -a exec        # Matches executive_summary
pidcast URL -a summ        # Matches summary
pidcast URL -a key         # Matches key_points
pidcast URL -m llama33     # Matches llama-3.3-70b-versatile
pidcast URL -m gpt120      # Matches openai/gpt-oss-120b
```

### 4. Typo? Helpful suggestions!
```bash
$ pidcast URL -a execcccc
✗ Unknown analysis type: 'execcccc'.
Use -L to list all available types.
```

---

## Your Most Common Commands (Memorize These)

### Quick transcription → Obsidian
```bash
pidcast "URL" -o
```

### Quick transcription → Obsidian + verbose
```bash
pidcast "URL" -ov
```

### Custom analysis type
```bash
pidcast "URL" -o -a summ
```

### Different model
```bash
pidcast "URL" -o -m llama33
```

### Force re-transcription
```bash
pidcast "URL" -of
```

### All the bells and whistles
```bash
pidcast "URL" -o -a exec -m llama33 -fv
```

---

## The Cheat Sheet

### Short Flags (Memorize These 7)
```
-o    Save to Obsidian
-a    Analysis type (fuzzy matching!)
-m    Model name (fuzzy matching!)
-f    Force re-transcription
-v    Verbose output
-L    List analysis types
-M    List models
```

### Common Analysis Types (Use Fuzzy!)
```
exec      → executive_summary
summ      → summary
key       → key_points
action    → action_items
comp      → comprehensive
```

### Common Models (Use Aliases!)
```
llama33   → llama-3.3-70b-versatile
llama8    → llama-3.1-8b-instant
gpt120    → openai/gpt-oss-120b
gpt20     → openai/gpt-oss-20b
compound  → groq/compound
```

---

## Pro Tips

### Chain flags for maximum speed:
```bash
-ofv     # Obsidian + force + verbose
-ov      # Obsidian + verbose
```

### Don't remember something? Just ask:
```bash
pidcast -L       # What analysis types exist?
pidcast -M       # What models exist?
pidcast --help   # Show everything
```

### Typo? Close enough usually works:
- `exec`, `execut`, `executive` all match `executive_summary`
- `sum`, `summ`, `summary` all match `summary`
- `llama3`, `llama33` both match `llama-3.3-70b-versatile`

---

## Before & After Examples

### Example 1: Daily workflow
**Before:**
```bash
pidcast "https://youtube.com/watch?v=abc" --save_to_obsidian --analysis_type executive_summary --verbose
```

**After:**
```bash
pidcast "https://youtube.com/watch?v=abc" -o -a exec -v
```

**Saved:** 47 keystrokes

---

### Example 2: Different model
**Before:**
```bash
pidcast "URL" --save_to_obsidian --groq_model llama-3.3-70b-versatile
```

**After:**
```bash
pidcast "URL" -o -m llama33
```

**Saved:** 40 keystrokes

---

### Example 3: Just checking options
**Before:**
*Read documentation, search YAML files*

**After:**
```bash
pidcast -L   # See all analysis types
pidcast -M   # See all models
```

**Saved:** Hours of your life

---

## Your Old Commands Still Work!

Don't worry - everything you've already scripted still works:
```bash
# These all still work perfectly
pidcast "URL" --save_to_obsidian
pidcast "URL" --analysis_type executive_summary
pidcast "URL" --groq_model llama-3.3-70b-versatile

# But now you can also use the shortcuts
pidcast "URL" -o
pidcast "URL" -a exec
pidcast "URL" -m llama33
```

**100% backward compatible. Zero breaking changes.**

---

## Next Time You Use Pidcast

1. Try the short flags: `-o -a -m -v`
2. Use fuzzy matching: `exec`, `summ`, `llama33`
3. Explore options: `pidcast -L` and `pidcast -M`
4. Chain flags: `-ofv`
5. Enjoy typing 60% less!

---

**Happy transcribing! 🎙️**
