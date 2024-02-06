from typing import Dict, List


class OpenHermesPrompt:
    """OpenHermes prompt class

    Args:
        system_prompt (str): OpenHermes system prompt
        end_str (str, optional): end of the model answer. Defaults to "<|im_end|>".
    """

    def __init__(self, system_prompt: str, end_strs: List[str] = ["<|im_end|>"]):  # noqa: B006
        self.system_prompt = system_prompt
        self.SYSTEM = "<|im_start|>system"
        self.USER = "<|im_start|>user"
        self.CONTEXT = "CONTEXT"
        self.ASSISTANT = "<|im_start|>assistant"
        self.END = "<|im_end|>"
        self.END_STRS = end_strs

    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating one of self.END_STRS, 1 if model is done generating one of self.END_STRS
        """
        for end_str in self.END_STRS:
            for i in range(len(end_str) - 1, 0, -1):
                # model is generating end_str -> wait until it finishes but don't print anything
                if answer[-i:] == end_str[:i]:
                    return 0
            # model is done generating end_str -> saving answer
            if answer[-len(end_str) :] == end_str:
                return 1
        # model can keep generating
        return -1

    def prepare(self, question: str, context: str) -> str:
        """Prepare prompt
            <|im_start|>system
            {{ system_prompt }}
            <|im_start|>user
            {{ user_prompt }}<|im_end|>
            <|im_start|>assistant
            {{ assistant_prompt }}<|im_end|>

        Args:
            question (str): question for the model
            context (str): context for the model

        Returns:
            str: prompt
        """
        out = f"{self.SYSTEM}\n"
        out += f"{self.system_prompt}\n"
        out += f"{self.CONTEXT}:\n{context}\n"
        out += f"{self.USER}\n"
        out += f"{question}{self.END}\n"
        out += f"{self.ASSISTANT}\n"
        return out
