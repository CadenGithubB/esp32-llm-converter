ESP32-S3 On-Device LLM — Training & Converter Package
=======================================================


STEP 1 — Install dependencies (once)
--------------------------------------

    pip install -r requirements.txt

On GPU machines, install PyTorch for your CUDA version first:

    CUDA 12.1:  pip install torch --index-url https://download.pytorch.org/whl/cu121
    CUDA 12.4:  pip install torch --index-url https://download.pytorch.org/whl/cu124

Verify GPU (optional):  python -c "import torch; print(torch.cuda.is_available())"


STEP 2 — Train the model
--------------------------

Use the narrow2 preset. It fits ~6.9 MB in PSRAM and gives the best Q→A accuracy.

    python train_tiny_model.py \
        --preset narrow2 \
        --text hardwareone_qa.txt hardwareone_qa_comprehensive.txt \
               hardwareone_qa_expanded.txt hardwareone_qa_troubleshooting.txt \
               hardwareone_qa_v2.txt \
        --epochs 30 \
        --batch-size 4 \
        --out ./out_model

The script prints Q&A test output after training. Check that answers match
questions. If they don't, run more epochs or add more training data.

Optional — add negatives to suppress hallucinations (Phase 2 training):

    python train_tiny_model.py \
        --preset narrow2 \
        --text hardwareone_qa.txt hardwareone_qa_comprehensive.txt \
               hardwareone_qa_expanded.txt hardwareone_qa_troubleshooting.txt \
               hardwareone_qa_v2.txt \
        --negatives hardwareone_qa_negatives.txt \
        --epochs 30 \
        --batch-size 4 \
        --out ./out_model

Note: Training is Q&A boundary-aware. Each Q/A pair is its own training block.
Answers will not bleed into unrelated questions.

To check model size before training:

    python train_tiny_model.py --preset narrow2 --estimate-only


STEP 3 — Convert to model.bin (browser tool)
---------------------------------------------

3a. Open index.html in Chrome or Edge (Firefox works too).
    No server needed — open the file directly from disk.

3b. Drop the entire output folder (./out_model) onto the page, or use the
    file picker to select the files inside it.

3c. Select INT8 quantization, group size 128. These are the defaults.

3d. Click Convert. When finished, click Download.
    This saves model.bin.


STEP 4 — Copy to SD card
--------------------------

4a. Copy model.bin to the root of the SD card.

4b. Insert the SD card and power on the device.

4c. The LLM page in the web UI will show the model loading.
    Expected log output: [LLM] Model ready. PSRAM used: ~6900KB


STEP 5 — Use the LLM
----------------------

Ask questions in Q: / A: format from the web UI or serial console.
The firmware stops generation when the model tries to emit a second Q:,
preventing runaway output chains.


PRESET REFERENCE
-----------------

Name      Vocab   Layers   dim   hidden   Est. INT8 PSRAM
--------  ------  -------  ----  ------   -----------------
narrow2   4K      20       128   640      ~6.9 MB  <- recommended
narrow    4K       8       256   512      ~7.3 MB
stretch2  8K      20       128   640      ~8.1 MB  (tight, may OOM)
stretch   8K      18       128   512      ~7.4 MB
leaner    8K      15       128   512      ~6.0 MB
baseline  16K     12       128   512      ~6.4 MB
