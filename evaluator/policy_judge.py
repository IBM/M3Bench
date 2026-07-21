from __future__ import annotations

from utils import JudgeInput, JudgeOutput


DOMAIN_MAPPING = {
    "california_schools": "Education & Students",
    "student_club": "Education & Students",
    "student_loan": "Education & Students",
    "university": "Education & Students",
    "cs_semester": "Education & Students",
    "computer_student": "Education & Students",
    "college_completion": "Education & Students",
    "movie": "Movies, TV & Entertainment, Social Media",
    "movie_platform": "Movies, TV & Entertainment, Social Media",
    "movie_3": "Movies, TV & Entertainment, Social Media",
    "movies_4": "Movies, TV & Entertainment, Social Media",
    "simpson_episodes": "Movies, TV & Entertainment, Social Media",
    "disney": "Movies, TV & Entertainment, Social Media",
    "law_episode": "Movies, TV & Entertainment, Social Media",
    "movielens": "Movies, TV & Entertainment, Social Media",
    "social_media": "Movies, TV & Entertainment, Social Media",
    "talkingdata": "Movies, TV & Entertainment, Social Media",
    "music_tracker": "Movies, TV & Entertainment, Social Media",
    "music_platform_2": "Movies, TV & Entertainment, Social Media",
    "card_games": "Movies, TV & Entertainment, Social Media",
    "video_games": "Movies, TV & Entertainment, Social Media",
    "superhero": "Movies, TV & Entertainment, Social Media",
    "formula_1": "Sports & Athletics",
    "professional_basketball": "Sports & Athletics",
    "european_football_1": "Sports & Athletics",
    "european_football_2": "Sports & Athletics",
    "olympics": "Sports & Athletics",
    "ice_hockey_draft": "Sports & Athletics",
    "hockey": "Sports & Athletics",
    "soccer_2016": "Sports & Athletics",
    "retails": "Retail, Sales & Commerce",
    "retail_world": "Retail, Sales & Commerce",
    "retail_complains": "Retail, Sales & Commerce",
    "superstore": "Retail, Sales & Commerce",
    "regional_sales": "Retail, Sales & Commerce",
    "car_retails": "Retail, Sales & Commerce",
    "book_publishing_company": "Retail, Sales & Commerce",
    "sales": "Retail, Sales & Commerce",
    "sales_in_weather": "Retail, Sales & Commerce",
    "works_cycles": "Retail, Sales & Commerce",
    "restaurant": "Food, Restaurants & Beverage",
    "food_inspection": "Food, Restaurants & Beverage",
    "food_inspection_2": "Food, Restaurants & Beverage",
    "cookbook": "Food, Restaurants & Beverage",
    "beer_factory": "Food, Restaurants & Beverage",
    "craftbeer": "Food, Restaurants & Beverage",
    "menu": "Food, Restaurants & Beverage",
    "app_store": "Technology & Software",
    "codebase_community": "Technology & Software",
    "codebase_comments": "Technology & Software",
    "software_company": "Technology & Software",
    "image_and_language": "Technology & Software",
    "toxicology": "Health & Medicine",
    "synthea": "Health & Medicine",
    "mental_health_survey": "Health & Medicine",
    "thrombosis_prediction": "Health & Medicine",
    "genes": "Health & Medicine",
    "mondial_geo": "Geography & Demographics",
    "world": "Geography & Demographics",
    "address": "Geography & Demographics",
    "world_development_indicators": "Geography & Demographics",
    "airline": "Transportation & Mobility",
    "trains": "Transportation & Mobility",
    "bike_share_1": "Transportation & Mobility",
    "cars": "Transportation & Mobility",
    "shipping": "Transportation & Mobility",
    "financial": "Finance & Economics",
    "donor": "Finance & Economics",
    "coinmarketcap": "Finance & Economics",
    "debit_card_specializing": "Finance & Economics",
    "legislator": "Government, Law & Public Services, Human Capital",
    "chicago_crime": "Government, Law & Public Services, Human Capital",
    "shooting": "Government, Law & Public Services, Human Capital",
    "public_review_platform": "Government, Law & Public Services, Human Capital",
    "human_resources": "Government, Law & Public Services, Human Capital",
    "books": "Literature, Language & Publishing",
    "authors": "Literature, Language & Publishing",
    "shakespeare": "Literature, Language & Publishing",
    "language_corpus": "Literature, Language & Publishing",
    "citeseer": "Literature, Language & Publishing",
}


class PolicyAdherenceJudge:
    """Evaluate whether the agent followed document-retriever policy instructions."""

    def judge(self, inp: JudgeInput) -> JudgeOutput:
        if inp.additional_instructions == "":
            return JudgeOutput(score=1.0, explanation="No policy.")

        domain_group = DOMAIN_MAPPING.get(inp.domain)
        instructions = inp.additional_instructions.lower()

        if domain_group and domain_group in inp.additional_instructions:
            if "do not use document retrievers" in instructions:
                for tool in inp.pred_tool_calls:
                    if "query_" in tool["name"]:
                        return JudgeOutput(
                            score=0.0,
                            explanation=(
                                f"Document retriever tool invoked for domain {inp.domain} "
                                f"for additional policy {inp.additional_instructions}. "
                                "Document retriever utilization not valid."
                            ),
                        )
            elif "only using document retrievers" in instructions:
                for tool in inp.pred_tool_calls:
                    if "query_" not in tool["name"]:
                        return JudgeOutput(
                            score=0.0,
                            explanation=(
                                f"Tool except for document retriever invoked for domain {inp.domain} "
                                f"for additional policy {inp.additional_instructions}. "
                                "Only document retriever tools should be utilized."
                            ),
                        )
        elif inp.additional_instructions == "Use document retrievers to answer questions. Do not use any other type of tool.":
            for tool in inp.pred_tool_calls:
                if "query_" not in tool["name"]:
                    return JudgeOutput(
                        score=0.0,
                        explanation=(
                            f"Tool except for document retriever invoked for additional policy {inp.additional_instructions}. "
                            "Only document retriever tools should be utilized."
                        ),
                    )
        elif inp.additional_instructions == "Do not use document retrievers to answer questions. Use other types of tools.":
            for tool in inp.pred_tool_calls:
                if "query_" in tool["name"]:
                    return JudgeOutput(
                        score=0.0,
                        explanation=(
                            f"Document retriever tool invoked for additional policy {inp.additional_instructions}. "
                            "Document retriever utilization not valid."
                        ),
                    )

        return JudgeOutput(
            score=1.0,
            explanation=(
                "Valid tool usage policy execution. "
                f"Additional Instruction : {inp.additional_instructions} and Domain : {inp.domain}."
            ),
        )


PolicyAdheranceJudge = PolicyAdherenceJudge
