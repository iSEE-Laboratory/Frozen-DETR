## Frozen-DETR: Enhancing DETR with Image Understanding from Frozen Foundation Models

### 1 Model Zoo

#### 1.1 COCO dataset

<table>
  <thead>
    <tr>
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>epoch</th>
      <th>box AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Frozen-DETR (Co-DINO-4scale)</td>
      <td>R50</td>
      <td>12</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen-DETR (Co-DINO-4scale)</td>
      <td>R50</td>
      <td>24</td>
      <td>54.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Frozen-DETR (Co-DINO-4scale)<sup></td>
      <td>SwinB (22k)</td>
      <td>12</td>
      <td>57.6</td>
    </tr>
  </tbody>
</table>

#### 1.2 LVIS dataset

<table>
  <thead>
    <tr>
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>epoch</th>
      <th>setting</th>
      <th>box AP</th>
      <th>box APr</th>
      <th>box APc</th>
      <th>box APf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Frozen-DETR (DINO-4scale)</td>
      <td>R50</td>
      <td>24</td>
      <td>closed-set</td>
      <td>41.0</td>
      <td>31.7</td>
      <td>41.0</td>
      <td>45.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen-DETR (DINO-4scale)</td>
      <td>R50</td>
      <td>24</td>
      <td>open-vocabulary</td>
      <td>40.0</td>
      <td>23.8</td>
      <td>41.9</td>
      <td>44.9</td>
    </tr>
  </tbody>
</table>

- Notes: 
  - All the checkpoints and logs are be found in [huggingface](https://huggingface.co/fushh7/Frozen-DETR) and [modelscope](https://modelscope.cn/models/fushh7/Frozen-DETR/summary). Please download and place them in `Frozen-DETR-ckpt`.
  - Results on the LVIS dataset is slightly different from the ones in the paper due to modifying the code.


### 2 Usage

- Our environment (other environments may also work)
  - python: 3.9.7
  - Torch: 1.10.2 + cu10.2
  - Torchvision: 0.11.2 + cu10.2
  - mmcv: 1.7.1

- Train

  ```
  bash tools/dist_train.sh configs/co-detr/co_dino_4scale_r50_1x_coco.py 8 50000
  ```

- Test

  ```
  bash tools/dist_test.sh configs/co-detr/co_dino_4scale_r50_1x_coco.py ../Frozen-DETR-ckpt/co_dino_4scale_r50_1x_coco.pth 4 50000 --eval=bbox
  ```



