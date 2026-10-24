import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import app as backend


def make_png() -> bytes:
    image = Image.new("RGB", (16, 16), color=(120, 40, 80))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(backend.app)

    def test_health_reports_tracked_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_available"])
        self.assertEqual(
            response.json()["model_file"],
            "leukemia_cnn_lstm_200.keras",
        )
        self.assertEqual(response.json()["model_type"], "CNN-LSTM hybrid")
        self.assertEqual(response.json()["input_shape"], [None, 128, 128, 3])
        self.assertEqual(response.json()["output_shape"], [None, 5])

    def test_health_reports_model_load_errors(self):
        with patch.object(
            backend,
            "get_model",
            side_effect=RuntimeError("model file is missing"),
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["model_available"])
        self.assertEqual(response.json()["error"], "model file is missing")

    def test_backend_class_order_matches_sorted_training_directories(self):
        training_classes = sorted(
            path.name
            for path in (backend.PROJECT_DIR / "data" / "sample_200").iterdir()
            if path.is_dir()
        )

        self.assertEqual(backend.CLASS_NAMES, training_classes)

    def test_prediction_uses_trained_class_order_and_real_model_output(self):
        cases = [
            ([0.70, 0.10, 0.05, 0.05, 0.10], "Acute Lymphoblastic Leukemia"),
            ([0.05, 0.10, 0.20, 0.15, 0.50], "Normal & Reactive"),
        ]

        for scores, expected_class in cases:
            with self.subTest(expected_class=expected_class):
                fake_model = Mock()
                fake_model.predict.return_value = np.array([scores])

                with patch.object(backend, "get_model", return_value=fake_model):
                    response = self.client.post(
                        "/predict",
                        files={
                            "file": (
                                "sample.png",
                                make_png(),
                                "image/png",
                            )
                        },
                    )

                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertTrue(body["success"])
                self.assertEqual(body["predicted_class"], expected_class)
                self.assertEqual(
                    body["result"],
                    "Normal & Reactive"
                    if expected_class == "Normal & Reactive"
                    else "Leukemia Detected",
                )
                self.assertEqual(body["confidence"], round(max(scores) * 100, 2))
                self.assertEqual(set(body["class_probabilities"]), set(backend.CLASS_NAMES))
                self.assertAlmostEqual(sum(body["class_probabilities"].values()), 100, delta=0.1)
                fake_model.predict.assert_called_once()

    def test_invalid_image_is_rejected(self):
        response = self.client.post(
            "/predict",
            files={"file": ("invalid.jpg", b"not an image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid JPEG or PNG", response.json()["detail"])

    def test_missing_upload_is_rejected(self):
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 422)

    def test_chat_requires_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = self.client.post(
                "/chat",
                json={"message": "What are leukemia symptoms?", "history": []},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    def test_chat_forwards_free_form_question_and_conversation_history(self):
        assistant_answer = "A general educational answer from the AI provider."
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=assistant_answer)
                )
            ]
        )
        openai_client = Mock()
        openai_client.chat.completions.create.return_value = completion
        history = [
            {"role": "user", "content": "What are leukemia symptoms?"},
            {"role": "assistant", "content": "A general educational overview."},
        ]

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch.object(backend, "OpenAI", return_value=openai_client),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain that in simpler words.",
                    "history": history,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"success": True, "answer": assistant_answer})
        kwargs = openai_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], backend.OPENAI_MODEL)
        self.assertEqual(kwargs["messages"][-3]["content"], history[0]["content"])
        self.assertEqual(kwargs["messages"][-2]["content"], history[1]["content"])
        self.assertEqual(kwargs["messages"][-1]["content"], "Explain that in simpler words.")

    def test_chat_provider_failure_is_reported_without_fabricated_response(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch.object(
                backend,
                "OpenAI",
                side_effect=backend.OpenAIError("provider unavailable"),
            ),
        ):
            response = self.client.post(
                "/chat",
                json={"message": "What are leukemia symptoms?", "history": []},
            )
            unknown_response = self.client.post(
                "/chat",
                json={"message": "A question outside the fallback FAQ", "history": []},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("temporarily unavailable", response.json()["detail"])
        self.assertEqual(unknown_response.status_code, 502)

    def test_chat_provider_timeout_returns_clear_gateway_timeout(self):
        timeout_error = backend.APITimeoutError(request=Mock())
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch.object(backend, "OpenAI", side_effect=timeout_error),
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain CNN-LSTM simply.", "history": []},
            )

        self.assertEqual(response.status_code, 504)
        self.assertIn("timed out", response.json()["detail"])

    def test_cors_allows_deployed_frontend(self):
        response = self.client.options(
            "/chat",
            headers={
                "Origin": backend.DEFAULT_FRONTEND_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            backend.DEFAULT_FRONTEND_ORIGIN,
        )

    def test_frontend_uses_backend_routes_and_single_api_base_url(self):
        frontend = (
            backend.PROJECT_DIR / "frontend" / "index.html"
        ).read_text(encoding="utf-8")
        prediction_handler = frontend[
            frontend.index("async function predictImage()"):
            frontend.index("// ---------------- CHATBOT ----------------")
        ]

        self.assertEqual(frontend.count("https://leukemiaai-api.onrender.com"), 1)
        self.assertIn('onclick="predictImage()"', frontend)
        self.assertIn("async function predictImage()", frontend)
        self.assertIn("`${API_URL}/predict`", frontend)
        self.assertIn('formData.append("file", file)', frontend)
        self.assertIn("signal: controller.signal", prediction_handler)
        self.assertIn("setTimeout(() => controller.abort(), 120000)", prediction_handler)
        self.assertIn("finally", prediction_handler)
        self.assertIn('predictButton.textContent =\n            "Analyze Image"', prediction_handler)
        self.assertIn("`${API_URL}/chat`", frontend)
        self.assertIn("recordPrediction(file.name", frontend)
        self.assertIn("id=\"modelTypeText\"", frontend)
        self.assertNotIn("OPENAI_API_KEY", frontend)
        self.assertNotIn("127.0.0.1:8000", frontend)


if __name__ == "__main__":
    unittest.main()
