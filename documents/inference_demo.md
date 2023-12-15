### Config and run demo script
1. You can download the demo case from [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main)🤗 or [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link), or download the whole demo dataset  [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) and choose any demo case you want.
2. Please set CT path and Ground Truth path of the case in the [config_demo.json](https://github.com/BAAI-DCAI/SegVol/blob/main/config/config_demo.json).
3. After that, config the [inference_demo.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/inference_demo.sh) file for execution:

    - `$segvol_ckpt`: the path of SegVol's checkpoint (Download from [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main)🤗 or [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link)).

    - `$work_dir`: any path of folder you want to save the log files and visualizaion results.

4. Finally, you can control the **prompt type**, **zoom-in-zoom-out mechanism** and **visualizaion switch** [here](https://github.com/BAAI-DCAI/SegVol/blob/35f3ff9c943a74f630e6948051a1fe21aaba91bc/inference_demo.py#L208C11-L208C11).
5. Now, just run `bash script/inference_demo.sh` to infer your demo case.
