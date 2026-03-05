# pidcast architecture

Central reference for how pidcast modules are organized, how they depend on each other, and how the core workflows flow through the codebase.

## High-level component diagram

Five layer groups with inter-group dependency arrows.

```plantuml
@startuml
title pidcast - High-level architecture

skinparam componentStyle rectangle
skinparam defaultFontSize 12
skinparam packageBorderThickness 2

package "Orchestration + Entry" as orch #E8F4FD {
  [cli] #E8F4FD
  [workflow] #E8F4FD
  [sync] #E8F4FD
  [digest] #E8F4FD
}

package "LLM Analysis" as llm #F3E8FD {
  [analysis] #F3E8FD
  [model_selector] #F3E8FD
  [summarization] #F3E8FD
  [providers] #F3E8FD
}

package "Media Processing" as media #E8F5E9 {
  [download] #E8F5E9
  [transcription] #E8F5E9
  [diarization] #E8F5E9
  [chunking] #E8F5E9
}

package "Persistence + Output" as persist #FFF8E1 {
  [markdown] #FFF8E1
  [history] #FFF8E1
  [library] #FFF8E1
  [rss] #FFF8E1
}

package "Foundation" as found #F5F5F5 {
  [config] #F5F5F5
  [exceptions] #F5F5F5
  [utils] #F5F5F5
  [config_manager] #F5F5F5
}

orch -down-> llm
orch -down-> media
orch -down-> persist
llm -down-> found
media -down-> found
persist -down-> found
llm -down-> media : chunking
persist -down-> persist : rss

legend right
  | Color | Layer |
  | <#E8F4FD> | Orchestration + Entry |
  | <#F3E8FD> | LLM Analysis |
  | <#E8F5E9> | Media Processing |
  | <#FFF8E1> | Persistence + Output |
  | <#F5F5F5> | Foundation |
endlegend

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/RPHDRzim38Rl-XL4UnnoQBS2NJ4KtKUC7LAqm7ORWaAJ6KU8B1cKtCpTzjz7oNOjTKM0E7f8cYy_b1iV98M-DIBeO10wNIlf0wpWgsxsAuEFQ42Iskk0AlI4GlY3jfqasO9oRUSisl0j31nBx25jOp1pgN4dUnCgnqxw2U7iF3DsKXraWvySrKZVzreTB7eFvq9C5ZYvuzJe0ycWdOKtiBQ1XXEG7XnRu7HzMRshli1l0V136RsTIJmV7HrsnXqBw0Uh2b3hXXCiw6-MVREvXOzMci5hdv8Qqy9fzPPTfvnoicvd767hQZGF7Wqtn55XydtRIj9FgPx2qf5xrDo7x5Mvb5kijOHxSefRf6sJvBG9HkKNw_UZeDeThN6otiuuGkwVzOfqDsLTB1oVgLcuslUMvzGiC1TpppAr3sWLyapk-j3r8GdgHWDNK5MNwxDH4XTyYBAsCuvmpswEXe8P_PFa2qRUB-TSG-LwMoVXAVCk7jdn8dwcVN1sfsC58um8VobCJV05xeCs9HbZ7rgURuEqc5X1MhnLB6Xr7HUY16aa9PgQ8kBop2p95UFyIZQtiAHvx1Xr1VEC_WkPprV0tHF2O8FS6-8B7RYsF_3P6KVytCW1-PdOXsd_h_dtgvVksMzSpkYNNu_yDRmkqVnoPPStf6b6bzSMQN5BBOzkowpPoBMC5Gbnmz_yt_KF)

## Module dependency graph

Every module and its internal imports. Solid arrows = top-level imports, dashed arrows = lazy/runtime imports.

```plantuml
@startuml
title pidcast - Module dependency graph

skinparam componentStyle rectangle
skinparam defaultFontSize 11
skinparam arrowThickness 1
skinparam linetype ortho

' --- Foundation ---
package "Foundation" as found #F5F5F5 {
  [config] as cfg #F5F5F5
  [exceptions] as exc #F5F5F5
  [utils] as utl #F5F5F5
  [config_manager] as cmgr #F5F5F5
}

' --- Media Processing ---
package "Media Processing" as media #E8F5E9 {
  [download] as dl #E8F5E9
  [transcription] as tx #E8F5E9
  [diarization] as dia #E8F5E9
  [chunking] as chk #E8F5E9
}

' --- LLM Analysis ---
package "LLM Analysis" as llm #F3E8FD {
  [analysis] as ana #F3E8FD
  [model_selector] as ms #F3E8FD
  [summarization] as summ #F3E8FD
  [providers/\nclaude_provider] as cp #F3E8FD
}

' --- Persistence + Output ---
package "Persistence + Output" as persist #FFF8E1 {
  [markdown] as md #FFF8E1
  [history] as hist #FFF8E1
  [library] as lib #FFF8E1
  [rss] as rss #FFF8E1
}

' --- Orchestration ---
package "Orchestration + Entry" as orch #E8F4FD {
  [cli] as cli #E8F4FD
  [workflow] as wf #E8F4FD
  [sync] as syn #E8F4FD
  [digest] as dig #E8F4FD
  [discovery] as disc #E8F4FD
}

' --- Solid arrows: top-level imports ---

' download
dl --> cfg
dl --> exc
dl --> utl

' transcription
tx --> cfg
tx --> exc
tx --> utl

' diarization
dia --> exc

' analysis
ana --> chk
ana --> cfg
ana --> exc
ana --> ms

' model_selector
ms --> exc

' summarization
summ --> hist

' claude_provider
cp --> ana
cp --> cfg
cp --> exc

' markdown
md --> cfg
md --> utl

' library
lib --> cfg
lib --> exc
lib --> rss

' config_manager
cmgr --> cfg

' workflow
wf --> ana
wf --> cfg
wf --> dl
wf --> exc
wf --> md
wf --> tx
wf --> utl

' sync
syn --> ana
syn --> cfg
syn --> dl
syn --> exc
syn --> hist
syn --> lib
syn --> md
syn --> rss
syn --> tx
syn --> utl

' digest
dig --> hist
dig --> lib
dig --> summ

' cli (top-level)
cli --> cfg
cli --> exc
cli --> utl
cli --> wf

' --- Dashed arrows: lazy/runtime imports ---
cli ..> lib : lazy
cli ..> hist : lazy
cli ..> syn : lazy
cli ..> disc : lazy
cli ..> dig : lazy
cli ..> cmgr : lazy
lib ..> cmgr : lazy
utl ..> cfg : lazy

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/RLR1JXin4BtxAomuj1KA2BL8j8UAIYGd44ZqrbR8NJkxLhpsolOIbgh_tXdROtP3bKEUvysCP-mtJgv253wElM5HHwFue6KZGkGhVkla20Qf1cMbiit4Mo-6Zh6mqtOGNlIySVtWhBBn8KxWwbKJXMsDchb8jHMZYHi7Jlf5yVFp6IcyT_lldMvsLeN0vvJHLiLfKDpvs3d6tl7LQiKtRhHIHEqiBjaWcfre5JzwjHzn4VWMb_nuSu4V_eTn_gDnTglRNyWssvOuPDHpemOC3Oc5vPmTepQP6AEP4pdXOoyiLE1puhxrrUKlrNohf1Rythi6UjIsNLP-oARw-sGyNbzkBjQVS_tIxQrnGgQDf24IgUY53OtNgOd4n-Sv3xcyVX6LdULEZNIZXMClPzFjAbSxkBcvvT-iC5FGOLdzd4cL6zF32No43DUvRb7Ob1qMn2BNEwdCOr06TEFo4VPXxX36lb_MZfQvn-3TavRAXxEVjZ5YbEgHJBcVeNhNTkx1NOS8abRyXD-DSHZZigt_UQJsXan0pitcSdsUMuGITtWzkG59958TE3i_9QQR1I9dz6yl2WTuJlcGpmk-gxcMV-URJWMuzBTZiAHE-Df6FwN27J3fOZ_HnJH6vnCocWWqxvtVROtR9sw_dLDXias-XCdExLAti6cHLhjaGkEUL6aILvMjxJmuesL-2S8N7jsmCkf96QxXQV4nwmrSIVyCjBzQVSKX9WWJIn1c59qN4y5W6YYYG8mei4JCPeJXV90NKAHWXkfDYRhTAuQaXD6VS1ymT2bmreTvteMuMH8siYWJf0_Kp439ICC6177l0ajAKY835P97WQNF8Zg6WYCFmfY4C2Wl5R5uv5Xwu2WCQ18B0w5GSGMYHu7I4C8T2kmbeVXCgDI8EcEeCKf967CIXgG4CIlXT7gqW6OgXlq8Oc-4ONE2LGcePePAhWbfWGa9uuNbUzByVPNj1uRhUZi5OuM4SHl2-orDmRK8dNeT0oDUfZC_sgXxjPW4Z3mzJPNmx5PDwNqvi65l1wOqXMzixQ4fNNMnuMQ7DlmrJBPj3MLNy4S1_ady0m00)

## Module reference table

| Module | Layer | Key exports | Description |
|--------|-------|-------------|-------------|
| `config` | Foundation | `VideoInfo`, `TranscriptionStats`, `AnalysisResult`, `PromptTemplate`, `PromptsConfig`, `get_project_root()`, `get_config_dir()` | Configuration constants, dataclasses, path resolution, env loading |
| `exceptions` | Foundation | `PidcastError`, `DownloadError`, `TranscriptionError`, `AnalysisError`, `ConfigurationError`, `LibraryError`, `DiarizationError` | Custom exception hierarchy |
| `utils` | Foundation | `setup_logging()`, `sanitize_filename()`, `create_smart_filename()`, `find_existing_transcription()`, `save_statistics()`, `cleanup_temp_files()` | Logging, filename handling, JSON I/O, duplicate detection |
| `config_manager` | Foundation | `ConfigManager` | File-based config management for library settings |
| `download` | Media Processing | `download_audio()`, `build_download_strategies()`, `VideoDownloadInfo` | YouTube audio download via yt-dlp with fallback strategies |
| `transcription` | Media Processing | `run_whisper_transcription()`, `process_local_file()`, `resolve_whisper_model()` | Whisper.cpp integration for audio-to-text |
| `diarization` | Media Processing | `DiarizationSegment`, `run_diarization()`, `merge_whisper_with_diarization()` | Speaker diarization using pyannote.audio |
| `chunking` | Media Processing | `chunk_transcript()`, `needs_chunking()`, `estimate_tokens()`, `format_chunk_for_analysis()` | Semantic transcript chunking for long content |
| `model_selector` | LLM Analysis | `ModelSelector`, `ModelConfig`, `load_models_config()`, `with_retry()` | Model selection, rate limit handling, retry with backoff |
| `analysis` | LLM Analysis | `analyze_transcript_with_llm()`, `load_analysis_prompts()`, `parse_llm_json_response()`, `render_analysis_to_terminal()` | Core LLM analysis pipeline via Groq API with chunking |
| `summarization` | LLM Analysis | `Summarizer` | Episode and digest summarization via Groq |
| `providers/claude_provider` | LLM Analysis | `analyze_with_claude_cli()`, `CLAUDE_MODELS` | Claude CLI provider (alternative to Groq) |
| `markdown` | Persistence + Output | `create_markdown_file()`, `create_analysis_markdown_file()`, `format_yaml_front_matter()` | Markdown output with YAML front matter for Obsidian |
| `history` | Persistence + Output | `ProcessingHistory`, `HistoryEntry`, `ProcessingStatus` | Processing history tracking with JSON persistence |
| `library` | Persistence + Output | `LibraryManager`, `Show` | Podcast library management with YAML persistence |
| `rss` | Persistence + Output | `RSSParser`, `Episode` | RSS feed parsing with retry logic |
| `cli` | Orchestration + Entry | `main()`, `DuplicateAction` | CLI with argparse, duplicate detection, library commands |
| `workflow` | Orchestration + Entry | `process_input_source()`, `run_analyze_existing_mode()`, `parse_transcript_file()` | Core workflow orchestration |
| `sync` | Orchestration + Entry | `SyncEngine` | Automated podcast sync (download + transcribe + analyze) |
| `digest` | Orchestration + Entry | `DigestGenerator`, `DigestFormatter`, `DigestData` | Multi-tier digest generation with topic clustering |
| `discovery` | Orchestration + Entry | `discover_podcast()`, `search_itunes_api()`, `prompt_user_selection()` | Podcast discovery via Apple Podcasts and iTunes API |
| `__init__` | Entry | `__version__` | Package init with version |
| `__main__` | Entry | - | `python -m pidcast` entry point |

## Core workflow sequence diagrams

### Single episode transcription

`process_input_source()` - the primary flow for transcribing a URL or local file.

```plantuml
@startuml
title Single episode transcription
hide footbox

participant "cli" as cli #E8F4FD
participant "workflow" as wf #E8F4FD
participant "download" as dl #E8F5E9
participant "transcription" as tx #E8F5E9
participant "diarization" as dia #E8F5E9
participant "markdown" as md #FFF8E1
participant "analysis" as ana #F3E8FD
participant "utils" as utl #F5F5F5

cli -> wf : process_input_source(url, args)
activate wf

wf -> dl : download_audio(url)
activate dl
dl --> wf : wav_path, video_info
deactivate dl

wf -> tx : run_whisper_transcription(wav_path)
activate tx
tx --> wf : transcript_text
deactivate tx

opt diarization enabled
  wf -> dia : run_diarization(wav_path)
  activate dia
  dia --> wf : segments
  deactivate dia
  wf -> dia : merge_whisper_with_diarization()
end

wf -> md : create_markdown_file(transcript, video_info)
activate md
md --> wf : transcript_path
deactivate md

opt analysis enabled
  wf -> ana : analyze_transcript_with_llm(transcript)
  activate ana
  ana --> wf : AnalysisResult
  deactivate ana
  wf -> md : create_analysis_markdown_file(result)
end

wf -> utl : save_statistics()
wf -> utl : cleanup_temp_files()

wf --> cli : transcript_path
deactivate wf

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/VPJ1Rjim38RlUWgXlQH0UnYs0fqFGmSiVe3j0GJLeXEYiYH8L9prwKVPSIAdmU238ViZV_8NfTT4Eb8Uh20a2_8FkXs_863o1YH5xL8NCH1w9_R8dthlwSqVXGWSY1q6xKZUTHRlf4wIt_9--z9-Qt-jWT77zzxwSQB6_ZPa_EYirsQ2Z9sWv-ttDRIgQIBfU9iqg2D-w3F7wzlWeEDxqPwemSZxjcrVjb_Ma7RQ_asO9eWNJ7tbP5SjP48x8vcuWVQvF48KNvv-bCOR6QBl82M5BcHIoUVOmIP7-oXrtAK7eJl2WoPWMWYEu3Zsef6BEqfdWxv4LAonWg6dHMFK1nKqxH_bWJVDirRlXO4QFwLcynePirFZ7bE0g5R-RfP4bHGT1GUTfIwy8ZXIhSAay85ajH4Id7wpO8IKfzPuK-O2AggIbV9ID6fUbe2pU8BT08vI-GvNO9r_WBY3SuiZqdubzo30cSKH7e16TX4uarecG_LeONDfjVQricOmWgDlMLEQgQrXSh9ccQfFlfG9Q-RV7r1joboyjKDLpSeb3YbBLxdqyoJo6rAsjFPgnZztlXHsPK6SKgpyAeFEEw4FeFWs8Kny6X9RMl_jB6YN0q_84AP41PW8HihX-AzPvI2yiYBVK_y0)

### Analyze existing transcript

`run_analyze_existing_mode()` - re-analyze an already-transcribed file.

```plantuml
@startuml
title Analyze existing transcript
hide footbox

participant "cli" as cli #E8F4FD
participant "workflow" as wf #E8F4FD
participant "analysis" as ana #F3E8FD
participant "markdown" as md #FFF8E1

cli -> wf : run_analyze_existing_mode(file_path)
activate wf

wf -> wf : parse_transcript_file(file_path)
note right: Extract text +\nYAML front matter

wf -> ana : load_analysis_prompts()
wf -> ana : analyze_transcript_with_llm(transcript)
activate ana
ana --> wf : AnalysisResult
deactivate ana

wf -> ana : render_analysis_to_terminal(result)

opt save to obsidian
  wf -> md : create_analysis_markdown_file(result)
end

wf --> cli : AnalysisResult
deactivate wf

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/VP8_QyCm4CLtVOeZMHnAXj8EmKD9eF7KBjqA1Q5OSYoYFqQwn6a_VKzsXUDIEXcPzxltdaxRY2BWsHg66es2dHFcyqk1kkg8sXq1Wt2n3hf3rcgfeF4U3_xAM4UShdKd7CAYDde18W9zORdVL8_LyrpG-t1gZEy7LT_yBHB9FEeuYEW0o-g1TBzKLeIJzBqRL5QIgAes-tl6aldwAKql8PmT5sCNdhjmwwKg6cqKxmIsAoPgr1U1YX362CimsKN5f-OyCRUWymG5VMonXFsLX3K2gYl2tOTxtxs-G1CyHRK2KOKyEXKgmNWXUIxAk-1jXx5OpJGv-4s2NcFBZR75zEycFm4ieUlSOFTZyARYsI2JQYQTcGNbf0fJ9FISGbjDvo8C-8ento54SL607lmXQgc5Om3Z7Df12NLGD7uQaxSqtbqUH5wZEs5fNVybJKlP4a2Fyni0)

### Library sync

`SyncEngine.sync()` - automated sync of subscribed podcast feeds.

```plantuml
@startuml
title Library sync
hide footbox

participant "cli" as cli #E8F4FD
participant "SyncEngine" as se #E8F4FD
participant "LibraryManager" as lib #FFF8E1
participant "RSSParser" as rss #FFF8E1
participant "ProcessingHistory" as hist #FFF8E1
participant "download" as dl #E8F5E9
participant "transcription" as tx #E8F5E9
participant "markdown" as md #FFF8E1
participant "analysis" as ana #F3E8FD

cli -> se : sync()
activate se

se -> lib : list_shows()
activate lib
lib --> se : shows[]
deactivate lib

loop for each show
  se -> rss : parse_feed(show.feed_url)
  activate rss
  rss --> se : episodes[]
  deactivate rss

  se -> hist : load history
  se -> se : filter new episodes

  loop for each new episode
    se -> dl : download_audio(episode.url)
    activate dl
    dl --> se : wav_path, video_info
    deactivate dl

    se -> tx : run_whisper_transcription(wav_path)
    activate tx
    tx --> se : transcript
    deactivate tx

    se -> md : create_markdown_file(transcript, info)

    opt analysis enabled
      se -> ana : analyze_transcript_with_llm(transcript)
      activate ana
      ana --> se : AnalysisResult
      deactivate ana
    end

    se -> hist : record(episode, status)
  end
end

se --> cli : sync complete
deactivate se

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/RPHDQnin48Rl-XL3Sh4X2PIsaEuXf12R7be8yR4K8Q_6Nb5PMYIjr-wlxuoykz46Nsn9yymxcW_jSqmgfEvWHJB98lmqsw326UBPrQ8n6c7dVThwan0jWQOshN89RcfhRa15e7-uNJsklwnVvi2615PkRnncBk9rR8ZtIpcrnv1HQxPmkrwl7rUVvkpRPlEgGXom4EDrx3Nu6cCqRl_3nEJ3EUCDhQ_pslVEUgKpfcs-vzVLjpcKWdAn3gPDnhjCfjDryg32NvRCq45VZqdfsdCqCKEq8UepYRq8mGMzV-9wLRa9YwLGTJ97bP0EXI03cRb65Vt696FZ-rXIP1DilvzacFZzHsYSCS9wtr9x0z1vaoa1SD7dubRGShNb3b4lsFh0Aza5koHkaYAKjkmm1SJMHA-HOm8KKHcTGkI6K0vK-RocHasshB8pDc40X_qaoDxpMnTMiet-rCKAnhvArMdZ5mFrCDo_o43Rl2UdAODU7MMhKdC7Hte3NXgtynS8IxSY98r31Q5pigTaMWnoDZ6BKV13w7JAUtAUGh_xVGn8S16GHgk2EY1Pv3Xpacg6YtU5E-2BBozklaqmpXsWKrkBEXj6HPx3weBym-B-iZUfaTOU2kNbu3bTZjp68vAPila-17p3sDaq44LEeniwNIOt34V0sWSzjkuEw6ELkiYncSy-p9C3FvlBWu7Q7rgB2SjvvvVpJ3nzwFu3)

### Digest generation

`DigestGenerator.generate_digest()` - create a multi-tier digest from processing history.

```plantuml
@startuml
title Digest generation
hide footbox

participant "cli" as cli #E8F4FD
participant "DigestGenerator" as dg #E8F4FD
participant "ProcessingHistory" as hist #FFF8E1
participant "LibraryManager" as lib #FFF8E1
participant "Summarizer" as summ #F3E8FD
participant "DigestFormatter" as fmt #E8F4FD

cli -> dg : generate_digest(since)
activate dg

dg -> hist : load recent entries
activate hist
hist --> dg : entries[]
deactivate hist

dg -> lib : list_shows()
activate lib
lib --> dg : shows[]
deactivate lib

dg -> summ : generate_one_liners(entries)
activate summ
summ --> dg : one-liners
deactivate summ

dg -> summ : generate_show_rollup(entries, show)
activate summ
summ --> dg : ShowRollup[]
deactivate summ

dg -> summ : generate_topic_clusters(entries)
activate summ
summ --> dg : TopicCluster[]
deactivate summ

dg -> fmt : format(DigestData)
activate fmt
fmt --> dg : markdown string
deactivate fmt

dg --> cli : digest path
deactivate dg

@enduml
```

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/ZPF1IyCm5CRl-Ik2krHm1z73w446RjM3WZXl8YLBi_PXcfJapJd_Ub_IhhQm3G-bJV9xt_Uzy3htA1nkQyqGK2k-W59vvAKooWa4QrW5XU8RQt5jlnbhY0O9ZJ38BwI62ouyfpUVB6VPRROO0wtQGojcNOIByZZxugnKteCf7y4Jl8zqHTzyacNPR7azvfzWxOJRFmiZIjLAQrWVPrVRkXOEVZhEqvB06ufnD71cNIqGEtfJOv-OXLQdTw69z719AYzYLKBPfRfaGY9yqJP1Z158UEmYvTgAWZib5LdHuq3vFpemB8BJWq77l7-mGesvJZSqJBAqa_lAxdmoSASp5ivxjKYCjGBJISKx6VHaZSerqC8dNOo1Te1PhEZLYP-s_D0WWYSSGfxSMQstpS7YAeOyRxGYuZLMZNivPuMs0PbBlVNuxuRUGirzMtBQAGn7Ihz769YadPw5G36G9e85gXUcIVmixCvmJobCELGER1GcDanQojlHuet0QWY6qPehKz2F-mi0)

## Configuration and data files

| File | Purpose |
|------|---------|
| `.env` / `.env.example` | API keys (GROQ_API_KEY), Obsidian path, whisper model path |
| `config/prompts.yaml` | LLM analysis prompt templates with JSON response format |
| `config/models.yaml` | LLM model definitions, rate limits, fallback chains |
| `data/transcripts/` | Generated transcript and analysis markdown files |
| `data/transcripts/transcription_stats.json` | Historical run statistics (duration, model, token counts) |
| `~/.config/pidcast/config.yaml` | Library settings (backfill limit, output dirs, cache hours) |
| `~/.config/pidcast/library.yaml` | Subscribed podcast shows |
| `~/.config/pidcast/history.json` | Processing history for sync engine |
