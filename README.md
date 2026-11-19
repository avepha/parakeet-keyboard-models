# parakeet-keyboard-models

Model files that [Parakeet Keyboard](https://github.com/avepha) downloads at runtime. The app fetches them itself and checks each file's size and SHA-256. Nothing here is app code.

## Pathumma STT FastConformer-RNNT Large (TH): fp32 ONNX for sherpa-onnx

Release: [`pathumma-stt-th-rnnt-large-fp32-v1`](https://github.com/avepha/parakeet-keyboard-models/releases/tag/pathumma-stt-th-rnnt-large-fp32-v1). Parakeet Keyboard uses it as its optional **More accurate** Thai voice model.

- **Original model:** Pathumma STT FastConformer-RNNT Large (TH) by **NECTEC** (National Electronics and Computer Technology Center, Thailand).
  - Source: https://huggingface.co/nectec/Pathumma-stt-th-fastconformer-rnnt-large at revision `ba2e5228a6281bece3e74d53e249cdfbefe74bfc`.
  - File: `pathumma-stt-th-fastconformer-rnnt-large.nemo`, sha256 `da2a54dd50e7a4cb6c7482737094f15282ae70db65c844708bae774204bb01a8`.
  - Authors: Wayupuk Sommuang, Siwakorn Kaewwichai, Sakson Bunta, Pattara Tipaksorn (2026).
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the same as the original. Full text in [`LICENSE`](LICENSE).
- **Changes:** converted to ONNX for sherpa-onnx's offline NeMo transducer (`model_type=nemo_transducer`) by avepha. The weights are unchanged: fp32, no quantization, no fine-tuning.
- **How the files were made:** the same steps as sherpa-onnx's own NeMo transducer exports (`scripts/nemo/` in [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)). The model is restored with NeMo, dither and pad_to are set to 0, `encoder.export()`, `decoder.export()` and `joint.export()` write the three ONNX files, and `tokens.txt` comes from `joint.vocabulary` plus `<blk>`. sherpa-onnx's metadata (vocab_size, normalize_type, pred_rnn_layers, pred_hidden, subsampling_factor 8) is written into the encoder.
- Not affiliated with or endorsed by NECTEC.

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `pathumma-rnnt-large-encoder.onnx` | 456050976 | `04733facba0c6feda1518c661f4ce2c9d0f2cf7cc28c92c96ad335339585955b` |
| `pathumma-rnnt-large-decoder.onnx` | 18374527 | `96a47876d13a5d5a69cb35923984b4b42c3f476dab28ea21d1ad45229943f5a8` |
| `pathumma-rnnt-large-joiner.onnx` | 8209571 | `e3f66e20d8e3d2391222ceeff738b151a595813a9b747f6871a755e3b452babd` |
| `pathumma-rnnt-large-tokens.txt` | 31888 | `4db0575d9abb9b053b4c1c4211865968e90c694fa4dfd9c21f0626d55b1dc94d` |

Using them with sherpa-onnx (Python):

```python
import sherpa_onnx
p = "pathumma-rnnt-large-"
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=p + "encoder.onnx", decoder=p + "decoder.onnx", joiner=p + "joiner.onnx",
    tokens=p + "tokens.txt", model_type="nemo_transducer", feature_dim=80, num_threads=4)
```

Citation (from the original model card):

```
@misc{Sommuang2026PathummaFastconRNNT,
    title     = { {Pathumma STT Fastconformer RNNT Large (TH)} },
    author    = { Wayupuk Sommuang and Siwakorn Kaewwichai and Sakson Bunta and Pattara Tipaksorn },
    url       = { https://huggingface.co/nectec/Pathumma-stt-th-fastconformer-rnnt-large },
    publisher = { Hugging Face },
    year      = { 2026 },
}
```
