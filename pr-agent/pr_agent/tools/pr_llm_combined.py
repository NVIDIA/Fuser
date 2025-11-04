import traceback
from functools import partial

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider_with_context
from pr_agent.log import get_logger
from pr_agent.tools.pr_description import PRDescription
from pr_agent.tools.pr_reviewer import PRReviewer


class PRActionCombined:
    def __init__(self, pr_url: str, args: list = None,
                 ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        """
        Initialize the PRDescription object with the necessary attributes and objects for generating a PR description
        using an AI model.
        Args:
            pr_url (str): The URL of the pull request.
            args (list, optional): List of arguments passed to the PRDescription class. Defaults to None.
        """
        self._pr_url = pr_url
        self._args = args
        self._ai_handler = ai_handler

        self.git_provider = get_git_provider_with_context(pr_url)

    async def run(self):
        try:
            # get_settings().pr_description.enable_help_comment = True
            descr = PRDescription(pr_url=self._pr_url, args=self._args, ai_handler=self._ai_handler)
            pr_description = await descr.run()

            review = PRReviewer(pr_url=self._pr_url, args=self._args, ai_handler=self._ai_handler, use_three_hash_header=True)
            pr_review = await review.run()

            internal_pr_review_placeholder = "<!-- BEGIN INTERNAL PR REVIEW PLACEHOLDER -->\n<!-- END INTERNAL PR REVIEW PLACEHOLDER -->\n\n"
            ci_test_results_placeholder = "<!-- BEGIN CI TEST RESULTS PLACEHOLDER -->\n<!-- END CI TEST RESULTS PLACEHOLDER -->\n\n"

            combined_output_dnr = '\n'.join([pr_description, pr_review, internal_pr_review_placeholder, ci_test_results_placeholder])
            initial_header="<!-- llm action combined persistent comment, dont modify this line -->"

            if get_settings().config.publish_combined_output:
                self.git_provider.publish_persistent_comment(initial_header + '\n\n' + combined_output_dnr,
                                                              initial_header=initial_header,
                                                              update_header=True,
                                                              name="review",
                                                              final_update_message=get_settings().pr_reviewer.final_update_message)
                get_settings().data = {}
            else:
                get_settings().data = {"artifact": combined_output_dnr}

        except Exception as e:
            get_logger().error(f"Error generating PR description {self.pr_id}: {e}")
            traceback.print_exc()
