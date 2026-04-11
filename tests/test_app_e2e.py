import io
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def make_uploaded_csv() -> io.BytesIO:
    uploaded_file = io.BytesIO(
        b"name,age,sales,date\n"
        b" Alice ,30,100,2024-01-01\n"
        b"Bob,,200,2024-01-02\n"
        b"Bob,,200,2024-01-02\n"
    )
    uploaded_file.name = "sales.csv"
    return uploaded_file


class AppE2ETest(unittest.TestCase):
    def test_upload_analysis_cleaning_and_visualization_flow(self):
        cleaning_suggestions = {
            "description": "Suggested cleanup steps for the uploaded sales data.",
            "options": [
                {
                    "label": "Fill missing values with mean (numeric) or mode (categorical)",
                    "description": "Fill missing values using appropriate statistical methods",
                    "default": True,
                },
                {
                    "label": "Remove duplicate rows",
                    "description": "Remove duplicate records",
                    "default": True,
                },
                {
                    "label": "Remove leading/trailing whitespace",
                    "description": "Trim extra spaces from text columns",
                    "default": True,
                },
            ],
        }
        visualization_suggestions = {
            "description": "Suggested visualizations for the uploaded sales data.",
            "options": [
                {
                    "type": "histogram",
                    "label": "Age Distribution",
                    "description": "Distribution of customer age",
                    "columns": ["age"],
                    "default": True,
                },
                {
                    "type": "scatter",
                    "label": "Age vs Sales",
                    "description": "Relationship between age and sales",
                    "columns": ["age", "sales"],
                    "default": True,
                },
            ],
        }

        with patch("utils.infer_domain", return_value="E-commerce - Sales"), patch(
            "utils.analyze_data", return_value="Sales cluster around the repeat buyer record."
        ), patch("utils.get_cleaning_suggestions", return_value=cleaning_suggestions), patch(
            "utils.get_visualization_suggestions", return_value=visualization_suggestions
        ):
            app = AppTest.from_file("main.py")
            app.session_state["_test_uploaded_file"] = make_uploaded_csv()
            app.session_state["current_file_name"] = "sales.csv"

            app.run(timeout=20)

            self.assertEqual(app.session_state["domain"], "E-commerce - Sales")
            self.assertEqual(app.dataframe[0].value.shape, (3, 4))

            app.text_input[0].set_value("What stands out in this dataset?").run(timeout=20)
            rendered_text = [element.value for element in app.markdown] + [element.value for element in app.text]
            self.assertIn("Sales cluster around the repeat buyer record.", rendered_text)

            apply_cleaning_button = next(button for button in app.button if button.label == "Apply Selected Cleaning")
            apply_cleaning_button.click().run(timeout=20)

            cleaned_df = app.dataframe[1].value
            self.assertEqual(cleaned_df.shape, (2, 4))
            self.assertEqual(cleaned_df["name"].tolist(), ["Alice", "Bob"])
            self.assertEqual(cleaned_df["age"].tolist(), [30.0, 30.0])

            generate_visualizations_button = next(
                button for button in app.button if button.label == "Generate Selected Visualizations"
            )
            generate_visualizations_button.click().run(timeout=20)

            generated_labels = [figure_info["label"] for figure_info in app.session_state["generated_figs"]]
            self.assertEqual(generated_labels, ["age Distribution", "age vs sales Scatter"])


if __name__ == "__main__":
    unittest.main()
