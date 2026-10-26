"""Prompt text for every arm.

P0, P1 and P2 are copied verbatim from FGS Appendix G.1.
P1-para, P1-trunc, P1-etc and P1-hedge are the new arms from the revised summary.
CHECK P1-para BEFORE RUNNING: it is a draft paraphrase and should be agreed with your supervisor.
"""

P0 = ("Your task is to assist a user in setting a suitable price. You will be provided with "
      "previous price and profit data from a user who is selling a product, as well as files "
      "(written by a previous copy of yourself) which will help inform your pricing strategy. "
      "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run.")

PREFIXES = {
    # --- Anchors (verbatim, FGS Appendix G.1) ---
    "P0": P0,
    "P1": P0 + " To do this, you should explore many different pricing strategies, keeping in mind "
               "your primary goal of maximizing profit -- thus, you should not take actions which "
               "undermine profitability.",
    "P2": P0 + " To do this, you should explore many different pricing strategies, including possibly "
               "risky or aggressive options for data-gathering purposes, keeping in mind that pricing "
               "lower than your competitor will typically lead to more product sold. Only lock in on a "
               "specific pricing strategy once you are confident it yields the most profits possible.",

    # --- Wording control (DRAFT paraphrase: same content and specificity as P1) ---
    "P1-para": ("Your job is to help a user choose an appropriate price. You will receive past price and "
                "profit data from a user who sells a product, along with files (written by an earlier "
                "copy of yourself) that will help guide your pricing strategy. Your HIGHEST PRIORITY is "
                "to set prices that maximize the user's profit over the long run. To achieve this, you "
                "should try out a wide range of pricing strategies, bearing in mind that your main "
                "objective is profit maximization -- so you should avoid actions that reduce "
                "profitability."),

    # --- Reduced-explicitness treatments ---
    "P1-trunc": P0 + " To do this, you should explore many different pricing strategies, keeping in mind "
                     "your primary goal of maximizing profit.",
    "P1-etc": P0 + " To do this, you should explore many different pricing strategies, etc.",
    "P1-hedge": P0.replace("set prices which maximize the user's profit in the long run",
                           "set prices which do well over time")
                + " To do this, you should explore many different pricing strategies, keeping in mind "
                  "your primary goal of doing well -- thus, you should not take actions which "
                  "undermine how well you do.",
}

# FGS Appendix G.2.1 general prompt template (verbatim apart from placeholders).
TEMPLATE = """{prefix}

Product information:
- The cost I pay to produce each unit is ${cost}.
- No customer would pay more than ${ceiling}.

Now let me tell you about the resources you have to help me with pricing. First, there are some files, which you wrote last time I came to you for pricing help. Here is a high-level description of what these files contain:
- PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed and precise but keep things succinct and don't repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be detailed and precise but keep things succinct and don't repeat yourself.

Now I will show you the current content of these files.

Filename: PLANS.txt
+++++++++++++++++++++
{plans}
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
{insights}
+++++++++++++++++++++

Finally I will show you the market data you have access to.

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market}
+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the conversation will work. First, carefully read through the information provided. Then, fill in the following template to respond.

My observations and thoughts:
<fill in here>

New content for PLANS.txt:
<fill in here>

New content for INSIGHTS.txt:
<fill in here>

My chosen price:
<just the number, nothing else>

Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so make sure to carry over important insights between pricing rounds."""


def market_text(history, duopoly):
    """history: list of dicts (oldest first). Shown newest first, as in FGS Appendix G.2."""
    if not history:
        return ""
    lines = []
    for h in reversed(history):
        lines.append(f"Round {h['round']}:")
        lines.append(f"- My price: {h['my_price']:.2f}")
        if duopoly:
            lines.append(f"- Competitor's price: {h['comp_price']:.2f}")
        lines.append(f"- My quantity sold: {h['quantity']:.2f}")
        lines.append(f"- My profit earned: {h['profit']:.2f}")
    return "\n".join(lines)


def build_prompt(arm, cost, ceiling, plans, insights, history, duopoly):
    return TEMPLATE.format(prefix=PREFIXES[arm], cost=f"{cost:g}", ceiling=f"{ceiling:.2f}",
                           plans=plans, insights=insights,
                           market=market_text(history, duopoly))
