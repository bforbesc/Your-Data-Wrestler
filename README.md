# 🤼 Your Data Wrestler

> **Learning case study** · Data transformation · Data visualization · Generative AI · Agents
>
> A historical experiment with explanations, examples, and source code.

[📚 Learning hub](https://github.com/bforbesc/Data-AI-with-Forbes) · [🧭 Beginner guide](https://github.com/bforbesc/Data-AI-with-Forbes/blob/main/START_HERE.md)

## The idea before the tools

**How can an AI assistant help with a table, and how do we check its work?**

**Table and question → inspect with tools → propose an answer or change → verify**

| Tool in this project | What it does in the example |
|---|---|
| **Pandas** | Inspects the table and applies selected cleaning operations. |
| **Plotly** | Draws interactive charts. |
| **Strands Agents + OpenAI** | Connects a language model to data-inspection tools and explanations. |
| **Streamlit** | Presents the upload, questions, options, and results in a browser. |

A language model generating an explanation is **generative AI**. A program choosing inspection tools is an **agent**. Neither replaces checking the numbers. This app does not train a supervised prediction model on the uploaded table.

**Ask questions about a table, explore its problems, and check what an AI assistant suggests.**

You do not need to write code for the first activity. The goal is to learn how to question an AI answer and decide whether a proposed data change makes sense. This Streamlit experiment combines a data table, an AI model, and tools that can inspect the data.

Part of [Data & AI with Forbes](https://github.com/bforbesc/Data-AI-with-Forbes). Built by @bforbesc and Claudia; kept as a practical and historical teaching reference.

## What it does, and why

The app accepts files, summarizes their contents, answers questions in ordinary language, suggests cleaning steps, and creates charts. A **data agent** uses an AI model to choose tools such as “count missing values” or “inspect a column.” A convincing explanation is not enough: you should be able to check the underlying calculation.

## Start here: no coding

Try the [hosted app](https://your-data-wrestler.streamlit.app/) if available, or set up a local copy using the developer reference below. AI features require a valid OpenAI API key and incur usage costs. Keep keys out of GitHub and shared outputs. Uploaded content may be sent to an external model provider; the included sample contains only fictional data.

1. Open [the fictional practice orders](examples/practice_orders.csv) on GitHub and choose **Download raw file** near the top right. Keep the `.csv` extension.
2. Upload the file using **Choose a file**. Each row is meant to be one order. The columns contain an order identifier, product category, number of units, and revenue in euros.
3. In **Data Overview**, expect **12 rows and 4 columns**. Inspect the table before asking the AI anything.
4. Ask: **“How many duplicate rows are there, and which order is duplicated?”** Check the answer yourself using the file: order `A001` occurs twice.
5. Ask: **“How many revenue values are missing? Explain how you checked.”** There is one blank revenue value, for `A011`.
6. Before applying cleaning, untick the other proposed actions and select only **Remove duplicate rows**, if offered. Choose **Apply Selected Cleaning** and download the result using **Download Cleaned Data**. Expect **11 rows**, with the missing revenue still blank.

If the AI misses an issue, that is a result to investigate. Suggestions vary between calls, so an absent checkbox is not proof the problem is absent. Keep a copy of the original file. The cleaned preview/download is a separate result; upload the downloaded file to start a new analysis of it.

## Try it on paper first

The sample is deliberately small enough to inspect without an app. Revenue for orders `A001` through `A010` is €10, €20, …, €100. `A011` has a missing revenue, and `A001` is repeated once.

| Question | Your prediction |
|---|---|
| How many rows are in the original file? | |
| How many represent unique orders after removing the exact duplicate? | |
| What is the sum of the known revenue before and after removing the duplicate? | |
| Should missing revenue be replaced by zero, an average, or left blank? Why? | |

<details>
<summary>Check the counts and calculations</summary>

There are **12 rows**, **11 unique orders**, **1 exact duplicate row**, and **1 missing revenue value**. Known revenue sums to **€560** with the duplicate and **€550** without it. Neither is the complete revenue total because one order has no revenue recorded.

If you fill the blank with an average before removing the duplicate, the average of the known values is **€560 / 11 ≈ €50.91**. Removing the duplicate first gives **€550 / 10 = €55**. Neither is a known sale. The order of operations and the assumption behind filling a blank both matter. “Missing” does not mean “zero.”

</details>

## Optional: install and look inside

The developer reference below explains setup and features. Use Python 3.11, matching `runtime.txt`, and a separate environment. The original project may need maintenance as dependencies evolve.

[main.py](main.py) builds the interface and applies selected cleaning actions. [utils.py](utils.py) contains data-inspection tools and AI calls. [prompts.py](prompts.py) contains instructions sent to the model. Start by finding the text of one button you used.

## License

Code, documentation, and the fictional teaching sample use the existing [MIT License](LICENSE). The sample contains no real transactions or personal information.

---

## Developer reference

For installation, the full feature list, and implementation details, see [the developer reference](docs/DEVELOPER_REFERENCE.md). The introductory activity above is the starting point for learners.
