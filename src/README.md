# Guided Querying over Videos using Autocompletion Suggestions

This repository accompanies the paper *Guided Querying over Videos using Autocompletion Suggestions*, showcasing the FastAPI demo used to explore text/emoji autocomplete for MSVD video retrieval. The server serves a single page (`/`) where users can type descriptions, preview autocomplete suggestions, and inspect the top retrieved clips.

*Abstract:* A critical challenge with querying video data is that the user is often unaware of the contents of the video, its structure, and the exact terminology to use in the query. While these problems exist in exploratory querying settings over traditional structured data, these problems are exacerbated for video data, where the information is sourced from human-annotated metadata or from computer vision models running over the video. In the absence of any guidance, the human is at a loss for where to begin the query session, or how to construct the query. Here, autocompletion-based user interfaces have become a popular and pervasive approach to interactive, keystroke-level query guidance. To guide the user through the query construction process, we develop methods that combine Vision Language Models and Large Language Models for generating query suggestions that are amenable to autocompletion-based user interfaces. Through quantitative assessments over real-world datasets, we demonstrate that our approach provides a meaningful benefit to query construction for video queries.

Project websites:

- Text Autocompletion: https://ixlab.github.io/vidautocomplete/
- Emoji Autocompletion: https://ixlab.github.io/emojiautocomplete/

## Environment setup

```bash
conda create -n msvd-autocomplete python=3.10 -y
conda activate msvd-autocomplete
pip install -r requirements.txt
```

### Downloading MSVD videos & thumbnails

```bash
wget https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar
tar -xvf YouTubeClips.tar
```

The archive expands into raw `YouTubeClips/` (`.avi`) files. Install `ffmpeg` inside your Conda environment before converting the clips:

```bash
conda install -c conda-forge ffmpeg
```

Then convert them to MP4 and capture the first frame as a thumbnail with the helper script:

```bash
mkdir -p static/msvd-vids static/msvd-imgs
python prepare_media.py \
	--source-dir /absolute/path/to/YouTubeClips \
	--videos-dir static/msvd-vids \
	--thumbnails-dir static/msvd-imgs
```

Useful flags:

- `--pattern` to target a different extension (defaults to `*.avi`).
- `--overwrite` to regenerate existing outputs.
- `--start-frame` to capture a different frame index for the thumbnail.
- `--video-codec` to pick a specific encoder (default `h264` so browsers can decode the MP4s; if your ffmpeg has libx264 built in, pass `--video-codec libx264` for better quality/size).
- `--workers` to process multiple videos in parallel (e.g., `--workers 8` to use 8 threads).

## Regenerating embeddings

If you do not have the `.npy` files yet, build them locally with the helper script. It encodes each description from the MSVD CSV using the same SentenceTransformer checkpoint as the server and saves one vector per video ID.

```bash
python generate_embeddings.py \
	--descriptions ./outputs/msvd-descriptions.csv \
	--output-dir ./embeddings/msvd
```

Key options:

- `--model`: sentence-transformers checkpoint (default `sentence-transformers/all-MiniLM-L6-v2`).
- `--batch-size`: encoding batch size (default `64`).
- `--force`: overwrite existing `.npy` files instead of skipping them.

Point the server to the generated directory via `MSVD_EMBEDDINGS_DIR` if you saved the files elsewhere.

## Data requirements

The backend expects the original CSV exports that ship with the main project:

- `outputs/msvd-phrases-emojis.csv`
- `outputs/msvd-descriptions.csv`

Embeddings (.npy files) are expected under `embeddings/msvd/` by default. If your embeddings live elsewhere, point the app to the folder via the `MSVD_EMBEDDINGS_DIR` environment variable.

Video media and thumbnails are served from `static/msvd-vids/` and `static/msvd-imgs/`. Create those folders (or symlink to your assets) so the browser can load the files returned by the `/search` endpoint.

## Running the server

Choose whether the autocomplete suggestions should include emojis via the `--mode` flag (default: `text`).

```bash
python app.py --mode text   # text-only suggestions
python app.py --mode emoji  # show phrase + emoji pairs
```

Common optional flags:

- `--host` (default `0.0.0.0`)
- `--port` (default `8080`)
- `--reload` to enable uvicorn auto-reload during development

Once the server starts, navigate to `http://<host>:<port>/` to use the demo. Start typing to see autocomplete suggestions, then click **Search** to compute the query embedding and retrieve the top matches. Each result row shows the thumbnail, a playable video preview, the truncated description, and the cosine similarity score.

## Cite this work

If you build on this demo, please cite the following:

```bibtex
@inproceedings{yoo2024guided,
	title={Guided Querying over Videos using Autocompletion Suggestions},
	author={Yoo, Hojin and Nandi, Arnab},
	booktitle={Proceedings of the 2024 Workshop on Human-In-the-Loop Data Analytics},
	pages={1--7},
	year={2024}
}
```
```bibtex
@inproceedings{yoo2025emojis,
	title={Emojis in Autocompletion: Enhancing Video Search with Visual Cues},
	author={Yoo, Hojin and Nandi, Arnab},
	booktitle={Proceedings of the 2025 Workshop on Human-In-the-Loop Data Analytics},
	year={2025}
}
```
