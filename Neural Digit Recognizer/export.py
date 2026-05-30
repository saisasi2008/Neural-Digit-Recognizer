"""
Multi-format model export: SavedModel, TFLite (quantized), and ONNX.
"""

import os
import numpy as np
import tensorflow as tf


def export_all(model: tf.keras.Model, x_test: np.ndarray = None,
               save_dir: str = "saved_model", quantize: bool = True, export_onnx: bool = True):
    """
    Export the trained model in multiple formats.

    Args:
        model: Trained Keras model.
        x_test: Test data for representative dataset (TFLite quantization).
        save_dir: Output directory.
        quantize: Whether to create quantized TFLite models.
        export_onnx: Whether to export to ONNX format.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print("  📦  MODEL EXPORT")
    print("═" * 60)

    sizes = {}

    # ── 1. Keras (.keras) ───────────────────────────────────────────────
    keras_path = os.path.join(save_dir, "mnist_advanced.keras")
    model.save(keras_path)
    sizes["Keras (.keras)"] = os.path.getsize(keras_path)
    print(f"  ✅ Keras:     {keras_path} ({_fmt_size(sizes['Keras (.keras)'])})")

    # ── 2. TFLite (Float32) ─────────────────────────────────────────────
    tflite_path = os.path.join(save_dir, "mnist_advanced.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    sizes["TFLite (Float32)"] = os.path.getsize(tflite_path)
    print(f"  ✅ TFLite:    {tflite_path} ({_fmt_size(sizes['TFLite (Float32)'])})")

    # ── 3. TFLite Quantized (INT8) ─────────────────────────────────────
    if quantize and x_test is not None:
        quant_path = os.path.join(save_dir, "mnist_advanced_int8.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data():
            for i in range(min(200, len(x_test))):
                yield [x_test[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        try:
            quant_model = converter.convert()
            with open(quant_path, "wb") as f:
                f.write(quant_model)
            sizes["TFLite (INT8)"] = os.path.getsize(quant_path)
            print(f"  ✅ TFLite Q:  {quant_path} ({_fmt_size(sizes['TFLite (INT8)'])})")
        except Exception as e:
            print(f"  ⚠️  INT8 quantization failed: {e}")

    # ── 4. TFLite Float16 ──────────────────────────────────────────────
    if quantize:
        fp16_path = os.path.join(save_dir, "mnist_advanced_fp16.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        try:
            fp16_model = converter.convert()
            with open(fp16_path, "wb") as f:
                f.write(fp16_model)
            sizes["TFLite (FP16)"] = os.path.getsize(fp16_path)
            print(f"  ✅ TFLite 16: {fp16_path} ({_fmt_size(sizes['TFLite (FP16)'])})")
        except Exception as e:
            print(f"  ⚠️  FP16 quantization failed: {e}")

    # ── 5. ONNX ────────────────────────────────────────────────────────
    if export_onnx:
        onnx_path = os.path.join(save_dir, "mnist_advanced.onnx")
        try:
            import tf2onnx
            import onnx
            spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
            sizes["ONNX"] = os.path.getsize(onnx_path)
            print(f"  ✅ ONNX:      {onnx_path} ({_fmt_size(sizes['ONNX'])})")
        except ImportError:
            print("  ⚠️  ONNX export skipped (install tf2onnx: pip install tf2onnx)")
        except Exception as e:
            print(f"  ⚠️  ONNX export failed: {e}")

    # ── Summary Table ──────────────────────────────────────────────────
    print("\n  📊  Model Size Comparison:")
    print("  " + "-" * 45)
    for fmt, size in sizes.items():
        bar_len = int(size / max(sizes.values()) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {fmt:<22} {bar} {_fmt_size(size)}")
    print("  " + "-" * 45)

    return sizes


def _fmt_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


if __name__ == "__main__":
    print("Loading saved model...")
    model = tf.keras.models.load_model("saved_model/mnist_advanced.keras")

    # Load test data for quantization
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    export_all(model, x_test)
