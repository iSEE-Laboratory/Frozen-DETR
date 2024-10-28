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
      <td>Frozen-DETR (DINO-det-4scale)</td>
      <td>R50</td>
      <td>12</td>
      <td>51.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen-DETR (DINO-det-4scale)</td>
      <td>R50</td>
      <td>24</td>
      <td>53.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Frozen-DETR (DINO-det-4scale+CLIP+DINOv2)<sup></td>
      <td>R50</td>
      <td>12</td>
      <td>53.8</td>
    </tr>
  </tbody>
</table>

- Notes: 
  - All the checkpoints and logs are be found in [huggingface](https://huggingface.co/fushh7/Frozen-DETR) and [modelscope](https://modelscope.cn/models/fushh7/Frozen-DETR/summary). Please download and place them in `Frozen-DETR-ckpt`.


### 2 Usage

- Our environment (other environments may also work)
  - python: 3.9.7
  - Torch: 1.10.2 + cu10.2
  - Torchvision: 0.11.2 + cu10.2
  - mmcv: 1.7.1

- Train

  ```
  bash scripts/DINO_train_dist.sh
  ```

- Test

  ```
  bash scripts/DINO_eval_dist.sh
  ```



