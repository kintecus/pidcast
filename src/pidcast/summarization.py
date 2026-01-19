"""LLM-based summarization for digest generation."""

import json
import logging
import re
from pathlib import Path

import yaml
from groq import Groq

from .history import HistoryEntry

logger = logging.getLogger(__name__)


class Summarizer:
    """LLM-based summarization for digest generation."""

    def __init__(self, prompts_path: Path, groq_api_key: str):
        """Initialize Summarizer.

        Args:
            prompts_path: Path to prompts YAML file
            groq_api_key: Groq API key
        """
        with open(prompts_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            self.prompts = data.get("prompts", {})

        self.groq_client = Groq(api_key=groq_api_key)

    def generate_one_liners(
        self, episodes: list[HistoryEntry], batch_size: int = 15
    ) -> dict[str, str]:
        """Generate one-line summaries for episodes in batch.

        Args:
            episodes: List of history entries to summarize
            batch_size: Number of episodes per API call

        Returns:
            Dict mapping episode GUID to one-liner
        """
        all_liners = {}

        # Load episode data
        episode_data = []
        for episode in episodes:
            if not episode.output_file:
                # Fallback to title if no output file
                all_liners[episode.guid] = episode.episode_title
                continue

            # Read markdown file for analysis content
            try:
                analysis_text = self._extract_analysis(episode.output_file)
                episode_data.append(
                    {
                        "guid": episode.guid,
                        "title": episode.episode_title,
                        "analysis": analysis_text[:1000],  # Truncate for context limit
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read analysis for {episode.episode_title}: {e}")
                all_liners[episode.guid] = episode.episode_title

        # Batch process
        for i in range(0, len(episode_data), batch_size):
            batch = episode_data[i : i + batch_size]

            try:
                prompt_template = self.prompts.get("one_liner_batch", {})
                prompt = prompt_template.get("user_prompt", "").format(
                    episodes=yaml.dump(batch)
                )
                system_prompt = prompt_template.get("system_prompt", "")

                response = self._call_groq_api(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model="llama-3.1-8b-instant",  # Fast model for simple task
                    json_mode=True,
                )

                # Parse JSON response: [{"guid": "...", "one_liner": "..."}, ...]
                if isinstance(response, list):
                    for item in response:
                        if "guid" in item and "one_liner" in item:
                            all_liners[item["guid"]] = item["one_liner"]
                else:
                    logger.warning(f"Unexpected response format for batch: {response}")
                    # Fallback to titles for this batch
                    for ep in batch:
                        if ep["guid"] not in all_liners:
                            all_liners[ep["guid"]] = ep["title"]

            except Exception as e:
                logger.error(f"Failed to generate one-liners for batch: {e}")
                # Fallback to titles for this batch
                for ep in batch:
                    if ep["guid"] not in all_liners:
                        all_liners[ep["guid"]] = ep["title"]

        return all_liners

    def generate_show_rollup(self, show_title: str, one_liners: list[str]) -> str:
        """Generate show-level summary from episode one-liners.

        Args:
            show_title: Name of the show
            one_liners: List of episode one-liners

        Returns:
            Show rollup summary
        """
        try:
            prompt_template = self.prompts.get("show_rollup", {})
            prompt = prompt_template.get("user_prompt", "").format(
                show_title=show_title,
                episode_count=len(one_liners),
                one_liners="\n".join(f"- {liner}" for liner in one_liners),
            )
            system_prompt = prompt_template.get("system_prompt", "")

            response = self._call_groq_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model="llama-3.1-8b-instant",
                json_mode=False,  # Plain text response
            )

            return response.strip() if isinstance(response, str) else str(response)

        except Exception as e:
            logger.error(f"Failed to generate show rollup for {show_title}: {e}")
            return f"{len(one_liners)} new episodes"

    def generate_topic_clusters(self, episode_data: list[dict]) -> list[dict]:
        """Generate cross-show topic clusters.

        Args:
            episode_data: List of {guid, show_id, title, one_liner}

        Returns:
            List of clusters: [{"topic": "...", "description": "...", "episode_guids": [...]}, ...]
        """
        try:
            prompt_template = self.prompts.get("topic_clustering", {})
            prompt = prompt_template.get("user_prompt", "").format(
                episode_data=yaml.dump(episode_data)
            )
            system_prompt = prompt_template.get("system_prompt", "")

            response = self._call_groq_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model="llama-3.1-70b-versatile",  # More powerful model for clustering
                json_mode=True,
            )

            # Expected response: [{"topic": "...", "description": "...", "episode_guids": [...]}]
            if isinstance(response, list):
                return response
            else:
                logger.warning(f"Unexpected response format for topic clustering: {response}")
                return []

        except Exception as e:
            logger.error(f"Failed to generate topic clusters: {e}")
            return []

    def _extract_analysis(self, markdown_path: str) -> str:
        """Extract analysis section from episode markdown.

        Args:
            markdown_path: Path to markdown file

        Returns:
            Extracted analysis text
        """
        with open(markdown_path, encoding="utf-8") as f:
            content = f.read()

        # Find JSON block in markdown
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("analysis", "")
            except json.JSONDecodeError:
                pass

        return ""

    def _call_groq_api(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "llama-3.1-8b-instant",
        json_mode: bool = False,
        max_tokens: int = 2000,
    ) -> str | list | dict:
        """Call Groq API for text generation.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model to use
            json_mode: Enable JSON mode
            max_tokens: Maximum output tokens

        Returns:
            Response text or parsed JSON
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.groq_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if json_mode:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                raise

        return content
