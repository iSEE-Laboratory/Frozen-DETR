# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

import mmcv
from ..builder import LOSSES
from .utils import weight_reduce_loss
from .utils import weighted_loss

# fmt: off
LVIS_CATEGORY_IMAGE_COUNT = [{'id': 1, 'image_count': 64}, {'id': 2, 'image_count': 364}, {'id': 3, 'image_count': 1911}, {'id': 4, 'image_count': 149}, {'id': 5, 'image_count': 29}, {'id': 6, 'image_count': 26}, {'id': 7, 'image_count': 59}, {'id': 8, 'image_count': 22}, {'id': 9, 'image_count': 12}, {'id': 10, 'image_count': 28}, {'id': 11, 'image_count': 505}, {'id': 12, 'image_count': 1207}, {'id': 13, 'image_count': 4}, {'id': 14, 'image_count': 10}, {'id': 15, 'image_count': 500}, {'id': 16, 'image_count': 33}, {'id': 17, 'image_count': 3}, {'id': 18, 'image_count': 44}, {'id': 19, 'image_count': 561}, {'id': 20, 'image_count': 8}, {'id': 21, 'image_count': 9}, {'id': 22, 'image_count': 33}, {'id': 23, 'image_count': 1883}, {'id': 24, 'image_count': 98}, {'id': 25, 'image_count': 70}, {'id': 26, 'image_count': 46}, {'id': 27, 'image_count': 117}, {'id': 28, 'image_count': 41}, {'id': 29, 'image_count': 1395}, {'id': 30, 'image_count': 7}, {'id': 31, 'image_count': 1}, {'id': 32, 'image_count': 314}, {'id': 33, 'image_count': 31}, {'id': 34, 'image_count': 1905}, {'id': 35, 'image_count': 1859}, {'id': 36, 'image_count': 1623}, {'id': 37, 'image_count': 47}, {'id': 38, 'image_count': 3}, {'id': 39, 'image_count': 3}, {'id': 40, 'image_count': 1}, {'id': 41, 'image_count': 305}, {'id': 42, 'image_count': 6}, {'id': 43, 'image_count': 210}, {'id': 44, 'image_count': 36}, {'id': 45, 'image_count': 1787}, {'id': 46, 'image_count': 17}, {'id': 47, 'image_count': 51}, {'id': 48, 'image_count': 138}, {'id': 49, 'image_count': 3}, {'id': 50, 'image_count': 1470}, {'id': 51, 'image_count': 3}, {'id': 52, 'image_count': 2}, {'id': 53, 'image_count': 186}, {'id': 54, 'image_count': 76}, {'id': 55, 'image_count': 26}, {'id': 56, 'image_count': 303}, {'id': 57, 'image_count': 738}, {'id': 58, 'image_count': 1799}, {'id': 59, 'image_count': 1934}, {'id': 60, 'image_count': 1609}, {'id': 61, 'image_count': 1622}, {'id': 62, 'image_count': 41}, {'id': 63, 'image_count': 4}, {'id': 64, 'image_count': 11}, {'id': 65, 'image_count': 270}, {'id': 66, 'image_count': 349}, {'id': 67, 'image_count': 42}, {'id': 68, 'image_count': 823}, {'id': 69, 'image_count': 6}, {'id': 70, 'image_count': 48}, {'id': 71, 'image_count': 3}, {'id': 72, 'image_count': 42}, {'id': 73, 'image_count': 24}, {'id': 74, 'image_count': 16}, {'id': 75, 'image_count': 605}, {'id': 76, 'image_count': 646}, {'id': 77, 'image_count': 1765}, {'id': 78, 'image_count': 2}, {'id': 79, 'image_count': 125}, {'id': 80, 'image_count': 1420}, {'id': 81, 'image_count': 140}, {'id': 82, 'image_count': 4}, {'id': 83, 'image_count': 322}, {'id': 84, 'image_count': 60}, {'id': 85, 'image_count': 2}, {'id': 86, 'image_count': 231}, {'id': 87, 'image_count': 333}, {'id': 88, 'image_count': 1941}, {'id': 89, 'image_count': 367}, {'id': 90, 'image_count': 1922}, {'id': 91, 'image_count': 18}, {'id': 92, 'image_count': 81}, {'id': 93, 'image_count': 1}, {'id': 94, 'image_count': 1852}, {'id': 95, 'image_count': 430}, {'id': 96, 'image_count': 247}, {'id': 97, 'image_count': 94}, {'id': 98, 'image_count': 21}, {'id': 99, 'image_count': 1821}, {'id': 100, 'image_count': 16}, {'id': 101, 'image_count': 12}, {'id': 102, 'image_count': 25}, {'id': 103, 'image_count': 41}, {'id': 104, 'image_count': 244}, {'id': 105, 'image_count': 7}, {'id': 106, 'image_count': 1}, {'id': 107, 'image_count': 40}, {'id': 108, 'image_count': 40}, {'id': 109, 'image_count': 104}, {'id': 110, 'image_count': 1671}, {'id': 111, 'image_count': 49}, {'id': 112, 'image_count': 243}, {'id': 113, 'image_count': 2}, {'id': 114, 'image_count': 242}, {'id': 115, 'image_count': 271}, {'id': 116, 'image_count': 104}, {'id': 117, 'image_count': 8}, {'id': 118, 'image_count': 1758}, {'id': 119, 'image_count': 1}, {'id': 120, 'image_count': 48}, {'id': 121, 'image_count': 14}, {'id': 122, 'image_count': 40}, {'id': 123, 'image_count': 1}, {'id': 124, 'image_count': 37}, {'id': 125, 'image_count': 1510}, {'id': 126, 'image_count': 6}, {'id': 127, 'image_count': 1903}, {'id': 128, 'image_count': 70}, {'id': 129, 'image_count': 86}, {'id': 130, 'image_count': 7}, {'id': 131, 'image_count': 5}, {'id': 132, 'image_count': 1406}, {'id': 133, 'image_count': 1901}, {'id': 134, 'image_count': 15}, {'id': 135, 'image_count': 28}, {'id': 136, 'image_count': 6}, {'id': 137, 'image_count': 494}, {'id': 138, 'image_count': 234}, {'id': 139, 'image_count': 1922}, {'id': 140, 'image_count': 1}, {'id': 141, 'image_count': 35}, {'id': 142, 'image_count': 5}, {'id': 143, 'image_count': 1828}, {'id': 144, 'image_count': 8}, {'id': 145, 'image_count': 63}, {'id': 146, 'image_count': 1668}, {'id': 147, 'image_count': 4}, {'id': 148, 'image_count': 95}, {'id': 149, 'image_count': 17}, {'id': 150, 'image_count': 1567}, {'id': 151, 'image_count': 2}, {'id': 152, 'image_count': 103}, {'id': 153, 'image_count': 50}, {'id': 154, 'image_count': 1309}, {'id': 155, 'image_count': 6}, {'id': 156, 'image_count': 92}, {'id': 157, 'image_count': 19}, {'id': 158, 'image_count': 37}, {'id': 159, 'image_count': 4}, {'id': 160, 'image_count': 709}, {'id': 161, 'image_count': 9}, {'id': 162, 'image_count': 82}, {'id': 163, 'image_count': 15}, {'id': 164, 'image_count': 3}, {'id': 165, 'image_count': 61}, {'id': 166, 'image_count': 51}, {'id': 167, 'image_count': 5}, {'id': 168, 'image_count': 13}, {'id': 169, 'image_count': 642}, {'id': 170, 'image_count': 24}, {'id': 171, 'image_count': 255}, {'id': 172, 'image_count': 9}, {'id': 173, 'image_count': 1808}, {'id': 174, 'image_count': 31}, {'id': 175, 'image_count': 158}, {'id': 176, 'image_count': 80}, {'id': 177, 'image_count': 1884}, {'id': 178, 'image_count': 158}, {'id': 179, 'image_count': 2}, {'id': 180, 'image_count': 12}, {'id': 181, 'image_count': 1659}, {'id': 182, 'image_count': 7}, {'id': 183, 'image_count': 834}, {'id': 184, 'image_count': 57}, {'id': 185, 'image_count': 174}, {'id': 186, 'image_count': 95}, {'id': 187, 'image_count': 27}, {'id': 188, 'image_count': 22}, {'id': 189, 'image_count': 1391}, {'id': 190, 'image_count': 90}, {'id': 191, 'image_count': 40}, {'id': 192, 'image_count': 445}, {'id': 193, 'image_count': 21}, {'id': 194, 'image_count': 1132}, {'id': 195, 'image_count': 177}, {'id': 196, 'image_count': 4}, {'id': 197, 'image_count': 17}, {'id': 198, 'image_count': 84}, {'id': 199, 'image_count': 55}, {'id': 200, 'image_count': 30}, {'id': 201, 'image_count': 25}, {'id': 202, 'image_count': 2}, {'id': 203, 'image_count': 125}, {'id': 204, 'image_count': 1135}, {'id': 205, 'image_count': 19}, {'id': 206, 'image_count': 72}, {'id': 207, 'image_count': 1926}, {'id': 208, 'image_count': 159}, {'id': 209, 'image_count': 7}, {'id': 210, 'image_count': 1}, {'id': 211, 'image_count': 13}, {'id': 212, 'image_count': 35}, {'id': 213, 'image_count': 18}, {'id': 214, 'image_count': 8}, {'id': 215, 'image_count': 6}, {'id': 216, 'image_count': 35}, {'id': 217, 'image_count': 1222}, {'id': 218, 'image_count': 103}, {'id': 219, 'image_count': 28}, {'id': 220, 'image_count': 63}, {'id': 221, 'image_count': 28}, {'id': 222, 'image_count': 5}, {'id': 223, 'image_count': 7}, {'id': 224, 'image_count': 14}, {'id': 225, 'image_count': 1918}, {'id': 226, 'image_count': 133}, {'id': 227, 'image_count': 16}, {'id': 228, 'image_count': 27}, {'id': 229, 'image_count': 110}, {'id': 230, 'image_count': 1895}, {'id': 231, 'image_count': 4}, {'id': 232, 'image_count': 1927}, {'id': 233, 'image_count': 8}, {'id': 234, 'image_count': 1}, {'id': 235, 'image_count': 263}, {'id': 236, 'image_count': 10}, {'id': 237, 'image_count': 2}, {'id': 238, 'image_count': 3}, {'id': 239, 'image_count': 87}, {'id': 240, 'image_count': 9}, {'id': 241, 'image_count': 71}, {'id': 242, 'image_count': 13}, {'id': 243, 'image_count': 18}, {'id': 244, 'image_count': 2}, {'id': 245, 'image_count': 5}, {'id': 246, 'image_count': 45}, {'id': 247, 'image_count': 1}, {'id': 248, 'image_count': 23}, {'id': 249, 'image_count': 32}, {'id': 250, 'image_count': 4}, {'id': 251, 'image_count': 1}, {'id': 252, 'image_count': 858}, {'id': 253, 'image_count': 661}, {'id': 254, 'image_count': 168}, {'id': 255, 'image_count': 210}, {'id': 256, 'image_count': 65}, {'id': 257, 'image_count': 4}, {'id': 258, 'image_count': 2}, {'id': 259, 'image_count': 159}, {'id': 260, 'image_count': 31}, {'id': 261, 'image_count': 811}, {'id': 262, 'image_count': 1}, {'id': 263, 'image_count': 42}, {'id': 264, 'image_count': 27}, {'id': 265, 'image_count': 2}, {'id': 266, 'image_count': 5}, {'id': 267, 'image_count': 95}, {'id': 268, 'image_count': 32}, {'id': 269, 'image_count': 1}, {'id': 270, 'image_count': 1}, {'id': 271, 'image_count': 1844}, {'id': 272, 'image_count': 897}, {'id': 273, 'image_count': 31}, {'id': 274, 'image_count': 23}, {'id': 275, 'image_count': 1}, {'id': 276, 'image_count': 202}, {'id': 277, 'image_count': 746}, {'id': 278, 'image_count': 44}, {'id': 279, 'image_count': 14}, {'id': 280, 'image_count': 26}, {'id': 281, 'image_count': 1}, {'id': 282, 'image_count': 2}, {'id': 283, 'image_count': 25}, {'id': 284, 'image_count': 238}, {'id': 285, 'image_count': 592}, {'id': 286, 'image_count': 26}, {'id': 287, 'image_count': 5}, {'id': 288, 'image_count': 42}, {'id': 289, 'image_count': 13}, {'id': 290, 'image_count': 46}, {'id': 291, 'image_count': 1}, {'id': 292, 'image_count': 8}, {'id': 293, 'image_count': 34}, {'id': 294, 'image_count': 5}, {'id': 295, 'image_count': 1}, {'id': 296, 'image_count': 1871}, {'id': 297, 'image_count': 717}, {'id': 298, 'image_count': 1010}, {'id': 299, 'image_count': 679}, {'id': 300, 'image_count': 3}, {'id': 301, 'image_count': 4}, {'id': 302, 'image_count': 1}, {'id': 303, 'image_count': 166}, {'id': 304, 'image_count': 2}, {'id': 305, 'image_count': 266}, {'id': 306, 'image_count': 101}, {'id': 307, 'image_count': 6}, {'id': 308, 'image_count': 14}, {'id': 309, 'image_count': 133}, {'id': 310, 'image_count': 2}, {'id': 311, 'image_count': 38}, {'id': 312, 'image_count': 95}, {'id': 313, 'image_count': 1}, {'id': 314, 'image_count': 12}, {'id': 315, 'image_count': 49}, {'id': 316, 'image_count': 5}, {'id': 317, 'image_count': 5}, {'id': 318, 'image_count': 16}, {'id': 319, 'image_count': 216}, {'id': 320, 'image_count': 12}, {'id': 321, 'image_count': 1}, {'id': 322, 'image_count': 54}, {'id': 323, 'image_count': 5}, {'id': 324, 'image_count': 245}, {'id': 325, 'image_count': 12}, {'id': 326, 'image_count': 7}, {'id': 327, 'image_count': 35}, {'id': 328, 'image_count': 36}, {'id': 329, 'image_count': 32}, {'id': 330, 'image_count': 1027}, {'id': 331, 'image_count': 10}, {'id': 332, 'image_count': 12}, {'id': 333, 'image_count': 1}, {'id': 334, 'image_count': 67}, {'id': 335, 'image_count': 71}, {'id': 336, 'image_count': 30}, {'id': 337, 'image_count': 48}, {'id': 338, 'image_count': 249}, {'id': 339, 'image_count': 13}, {'id': 340, 'image_count': 29}, {'id': 341, 'image_count': 14}, {'id': 342, 'image_count': 236}, {'id': 343, 'image_count': 15}, {'id': 344, 'image_count': 1521}, {'id': 345, 'image_count': 25}, {'id': 346, 'image_count': 249}, {'id': 347, 'image_count': 139}, {'id': 348, 'image_count': 2}, {'id': 349, 'image_count': 2}, {'id': 350, 'image_count': 1890}, {'id': 351, 'image_count': 1240}, {'id': 352, 'image_count': 1}, {'id': 353, 'image_count': 9}, {'id': 354, 'image_count': 1}, {'id': 355, 'image_count': 3}, {'id': 356, 'image_count': 11}, {'id': 357, 'image_count': 4}, {'id': 358, 'image_count': 236}, {'id': 359, 'image_count': 44}, {'id': 360, 'image_count': 19}, {'id': 361, 'image_count': 1100}, {'id': 362, 'image_count': 7}, {'id': 363, 'image_count': 69}, {'id': 364, 'image_count': 2}, {'id': 365, 'image_count': 8}, {'id': 366, 'image_count': 5}, {'id': 367, 'image_count': 227}, {'id': 368, 'image_count': 6}, {'id': 369, 'image_count': 106}, {'id': 370, 'image_count': 81}, {'id': 371, 'image_count': 17}, {'id': 372, 'image_count': 134}, {'id': 373, 'image_count': 312}, {'id': 374, 'image_count': 8}, {'id': 375, 'image_count': 271}, {'id': 376, 'image_count': 2}, {'id': 377, 'image_count': 103}, {'id': 378, 'image_count': 1938}, {'id': 379, 'image_count': 574}, {'id': 380, 'image_count': 120}, {'id': 381, 'image_count': 2}, {'id': 382, 'image_count': 2}, {'id': 383, 'image_count': 13}, {'id': 384, 'image_count': 29}, {'id': 385, 'image_count': 1710}, {'id': 386, 'image_count': 66}, {'id': 387, 'image_count': 1008}, {'id': 388, 'image_count': 1}, {'id': 389, 'image_count': 3}, {'id': 390, 'image_count': 1942}, {'id': 391, 'image_count': 19}, {'id': 392, 'image_count': 1488}, {'id': 393, 'image_count': 46}, {'id': 394, 'image_count': 106}, {'id': 395, 'image_count': 115}, {'id': 396, 'image_count': 19}, {'id': 397, 'image_count': 2}, {'id': 398, 'image_count': 1}, {'id': 399, 'image_count': 28}, {'id': 400, 'image_count': 9}, {'id': 401, 'image_count': 192}, {'id': 402, 'image_count': 12}, {'id': 403, 'image_count': 21}, {'id': 404, 'image_count': 247}, {'id': 405, 'image_count': 6}, {'id': 406, 'image_count': 64}, {'id': 407, 'image_count': 7}, {'id': 408, 'image_count': 40}, {'id': 409, 'image_count': 542}, {'id': 410, 'image_count': 2}, {'id': 411, 'image_count': 1898}, {'id': 412, 'image_count': 36}, {'id': 413, 'image_count': 4}, {'id': 414, 'image_count': 1}, {'id': 415, 'image_count': 191}, {'id': 416, 'image_count': 6}, {'id': 417, 'image_count': 41}, {'id': 418, 'image_count': 39}, {'id': 419, 'image_count': 46}, {'id': 420, 'image_count': 1}, {'id': 421, 'image_count': 1451}, {'id': 422, 'image_count': 1878}, {'id': 423, 'image_count': 11}, {'id': 424, 'image_count': 82}, {'id': 425, 'image_count': 18}, {'id': 426, 'image_count': 1}, {'id': 427, 'image_count': 7}, {'id': 428, 'image_count': 3}, {'id': 429, 'image_count': 575}, {'id': 430, 'image_count': 1907}, {'id': 431, 'image_count': 8}, {'id': 432, 'image_count': 4}, {'id': 433, 'image_count': 32}, {'id': 434, 'image_count': 11}, {'id': 435, 'image_count': 4}, {'id': 436, 'image_count': 54}, {'id': 437, 'image_count': 202}, {'id': 438, 'image_count': 32}, {'id': 439, 'image_count': 3}, {'id': 440, 'image_count': 130}, {'id': 441, 'image_count': 119}, {'id': 442, 'image_count': 141}, {'id': 443, 'image_count': 29}, {'id': 444, 'image_count': 525}, {'id': 445, 'image_count': 1323}, {'id': 446, 'image_count': 2}, {'id': 447, 'image_count': 113}, {'id': 448, 'image_count': 16}, {'id': 449, 'image_count': 7}, {'id': 450, 'image_count': 35}, {'id': 451, 'image_count': 1908}, {'id': 452, 'image_count': 353}, {'id': 453, 'image_count': 18}, {'id': 454, 'image_count': 14}, {'id': 455, 'image_count': 77}, {'id': 456, 'image_count': 8}, {'id': 457, 'image_count': 37}, {'id': 458, 'image_count': 1}, {'id': 459, 'image_count': 346}, {'id': 460, 'image_count': 19}, {'id': 461, 'image_count': 1779}, {'id': 462, 'image_count': 23}, {'id': 463, 'image_count': 25}, {'id': 464, 'image_count': 67}, {'id': 465, 'image_count': 19}, {'id': 466, 'image_count': 28}, {'id': 467, 'image_count': 4}, {'id': 468, 'image_count': 27}, {'id': 469, 'image_count': 1861}, {'id': 470, 'image_count': 11}, {'id': 471, 'image_count': 13}, {'id': 472, 'image_count': 13}, {'id': 473, 'image_count': 32}, {'id': 474, 'image_count': 1767}, {'id': 475, 'image_count': 42}, {'id': 476, 'image_count': 17}, {'id': 477, 'image_count': 128}, {'id': 478, 'image_count': 1}, {'id': 479, 'image_count': 9}, {'id': 480, 'image_count': 10}, {'id': 481, 'image_count': 4}, {'id': 482, 'image_count': 9}, {'id': 483, 'image_count': 18}, {'id': 484, 'image_count': 41}, {'id': 485, 'image_count': 28}, {'id': 486, 'image_count': 3}, {'id': 487, 'image_count': 65}, {'id': 488, 'image_count': 9}, {'id': 489, 'image_count': 23}, {'id': 490, 'image_count': 24}, {'id': 491, 'image_count': 1}, {'id': 492, 'image_count': 2}, {'id': 493, 'image_count': 59}, {'id': 494, 'image_count': 48}, {'id': 495, 'image_count': 17}, {'id': 496, 'image_count': 1877}, {'id': 497, 'image_count': 18}, {'id': 498, 'image_count': 1920}, {'id': 499, 'image_count': 50}, {'id': 500, 'image_count': 1890}, {'id': 501, 'image_count': 99}, {'id': 502, 'image_count': 1530}, {'id': 503, 'image_count': 3}, {'id': 504, 'image_count': 11}, {'id': 505, 'image_count': 19}, {'id': 506, 'image_count': 3}, {'id': 507, 'image_count': 63}, {'id': 508, 'image_count': 5}, {'id': 509, 'image_count': 6}, {'id': 510, 'image_count': 233}, {'id': 511, 'image_count': 54}, {'id': 512, 'image_count': 36}, {'id': 513, 'image_count': 10}, {'id': 514, 'image_count': 124}, {'id': 515, 'image_count': 101}, {'id': 516, 'image_count': 3}, {'id': 517, 'image_count': 363}, {'id': 518, 'image_count': 3}, {'id': 519, 'image_count': 30}, {'id': 520, 'image_count': 18}, {'id': 521, 'image_count': 199}, {'id': 522, 'image_count': 97}, {'id': 523, 'image_count': 32}, {'id': 524, 'image_count': 121}, {'id': 525, 'image_count': 16}, {'id': 526, 'image_count': 12}, {'id': 527, 'image_count': 2}, {'id': 528, 'image_count': 214}, {'id': 529, 'image_count': 48}, {'id': 530, 'image_count': 26}, {'id': 531, 'image_count': 13}, {'id': 532, 'image_count': 4}, {'id': 533, 'image_count': 11}, {'id': 534, 'image_count': 123}, {'id': 535, 'image_count': 7}, {'id': 536, 'image_count': 200}, {'id': 537, 'image_count': 91}, {'id': 538, 'image_count': 9}, {'id': 539, 'image_count': 72}, {'id': 540, 'image_count': 1886}, {'id': 541, 'image_count': 4}, {'id': 542, 'image_count': 1}, {'id': 543, 'image_count': 1}, {'id': 544, 'image_count': 1932}, {'id': 545, 'image_count': 4}, {'id': 546, 'image_count': 56}, {'id': 547, 'image_count': 854}, {'id': 548, 'image_count': 755}, {'id': 549, 'image_count': 1843}, {'id': 550, 'image_count': 96}, {'id': 551, 'image_count': 7}, {'id': 552, 'image_count': 74}, {'id': 553, 'image_count': 66}, {'id': 554, 'image_count': 57}, {'id': 555, 'image_count': 44}, {'id': 556, 'image_count': 1905}, {'id': 557, 'image_count': 4}, {'id': 558, 'image_count': 90}, {'id': 559, 'image_count': 1635}, {'id': 560, 'image_count': 8}, {'id': 561, 'image_count': 5}, {'id': 562, 'image_count': 50}, {'id': 563, 'image_count': 545}, {'id': 564, 'image_count': 20}, {'id': 565, 'image_count': 193}, {'id': 566, 'image_count': 285}, {'id': 567, 'image_count': 3}, {'id': 568, 'image_count': 1}, {'id': 569, 'image_count': 1904}, {'id': 570, 'image_count': 294}, {'id': 571, 'image_count': 3}, {'id': 572, 'image_count': 5}, {'id': 573, 'image_count': 24}, {'id': 574, 'image_count': 2}, {'id': 575, 'image_count': 2}, {'id': 576, 'image_count': 16}, {'id': 577, 'image_count': 8}, {'id': 578, 'image_count': 154}, {'id': 579, 'image_count': 66}, {'id': 580, 'image_count': 1}, {'id': 581, 'image_count': 24}, {'id': 582, 'image_count': 1}, {'id': 583, 'image_count': 4}, {'id': 584, 'image_count': 75}, {'id': 585, 'image_count': 6}, {'id': 586, 'image_count': 126}, {'id': 587, 'image_count': 24}, {'id': 588, 'image_count': 22}, {'id': 589, 'image_count': 1872}, {'id': 590, 'image_count': 16}, {'id': 591, 'image_count': 423}, {'id': 592, 'image_count': 1927}, {'id': 593, 'image_count': 38}, {'id': 594, 'image_count': 3}, {'id': 595, 'image_count': 1945}, {'id': 596, 'image_count': 35}, {'id': 597, 'image_count': 1}, {'id': 598, 'image_count': 13}, {'id': 599, 'image_count': 9}, {'id': 600, 'image_count': 14}, {'id': 601, 'image_count': 37}, {'id': 602, 'image_count': 3}, {'id': 603, 'image_count': 4}, {'id': 604, 'image_count': 100}, {'id': 605, 'image_count': 195}, {'id': 606, 'image_count': 1}, {'id': 607, 'image_count': 12}, {'id': 608, 'image_count': 24}, {'id': 609, 'image_count': 489}, {'id': 610, 'image_count': 10}, {'id': 611, 'image_count': 1689}, {'id': 612, 'image_count': 42}, {'id': 613, 'image_count': 81}, {'id': 614, 'image_count': 894}, {'id': 615, 'image_count': 1868}, {'id': 616, 'image_count': 7}, {'id': 617, 'image_count': 1567}, {'id': 618, 'image_count': 10}, {'id': 619, 'image_count': 8}, {'id': 620, 'image_count': 7}, {'id': 621, 'image_count': 629}, {'id': 622, 'image_count': 89}, {'id': 623, 'image_count': 15}, {'id': 624, 'image_count': 134}, {'id': 625, 'image_count': 4}, {'id': 626, 'image_count': 1802}, {'id': 627, 'image_count': 595}, {'id': 628, 'image_count': 1210}, {'id': 629, 'image_count': 48}, {'id': 630, 'image_count': 418}, {'id': 631, 'image_count': 1846}, {'id': 632, 'image_count': 5}, {'id': 633, 'image_count': 221}, {'id': 634, 'image_count': 10}, {'id': 635, 'image_count': 7}, {'id': 636, 'image_count': 76}, {'id': 637, 'image_count': 22}, {'id': 638, 'image_count': 10}, {'id': 639, 'image_count': 341}, {'id': 640, 'image_count': 1}, {'id': 641, 'image_count': 705}, {'id': 642, 'image_count': 1900}, {'id': 643, 'image_count': 188}, {'id': 644, 'image_count': 227}, {'id': 645, 'image_count': 861}, {'id': 646, 'image_count': 6}, {'id': 647, 'image_count': 115}, {'id': 648, 'image_count': 5}, {'id': 649, 'image_count': 43}, {'id': 650, 'image_count': 14}, {'id': 651, 'image_count': 6}, {'id': 652, 'image_count': 15}, {'id': 653, 'image_count': 1167}, {'id': 654, 'image_count': 15}, {'id': 655, 'image_count': 994}, {'id': 656, 'image_count': 28}, {'id': 657, 'image_count': 2}, {'id': 658, 'image_count': 338}, {'id': 659, 'image_count': 334}, {'id': 660, 'image_count': 15}, {'id': 661, 'image_count': 102}, {'id': 662, 'image_count': 1}, {'id': 663, 'image_count': 8}, {'id': 664, 'image_count': 1}, {'id': 665, 'image_count': 1}, {'id': 666, 'image_count': 28}, {'id': 667, 'image_count': 91}, {'id': 668, 'image_count': 260}, {'id': 669, 'image_count': 131}, {'id': 670, 'image_count': 128}, {'id': 671, 'image_count': 3}, {'id': 672, 'image_count': 10}, {'id': 673, 'image_count': 39}, {'id': 674, 'image_count': 2}, {'id': 675, 'image_count': 925}, {'id': 676, 'image_count': 354}, {'id': 677, 'image_count': 31}, {'id': 678, 'image_count': 10}, {'id': 679, 'image_count': 215}, {'id': 680, 'image_count': 71}, {'id': 681, 'image_count': 43}, {'id': 682, 'image_count': 28}, {'id': 683, 'image_count': 34}, {'id': 684, 'image_count': 16}, {'id': 685, 'image_count': 273}, {'id': 686, 'image_count': 2}, {'id': 687, 'image_count': 999}, {'id': 688, 'image_count': 4}, {'id': 689, 'image_count': 107}, {'id': 690, 'image_count': 2}, {'id': 691, 'image_count': 1}, {'id': 692, 'image_count': 454}, {'id': 693, 'image_count': 9}, {'id': 694, 'image_count': 1901}, {'id': 695, 'image_count': 61}, {'id': 696, 'image_count': 91}, {'id': 697, 'image_count': 46}, {'id': 698, 'image_count': 1402}, {'id': 699, 'image_count': 74}, {'id': 700, 'image_count': 421}, {'id': 701, 'image_count': 226}, {'id': 702, 'image_count': 10}, {'id': 703, 'image_count': 1720}, {'id': 704, 'image_count': 261}, {'id': 705, 'image_count': 1337}, {'id': 706, 'image_count': 293}, {'id': 707, 'image_count': 62}, {'id': 708, 'image_count': 814}, {'id': 709, 'image_count': 407}, {'id': 710, 'image_count': 6}, {'id': 711, 'image_count': 16}, {'id': 712, 'image_count': 7}, {'id': 713, 'image_count': 1791}, {'id': 714, 'image_count': 2}, {'id': 715, 'image_count': 1915}, {'id': 716, 'image_count': 1940}, {'id': 717, 'image_count': 13}, {'id': 718, 'image_count': 16}, {'id': 719, 'image_count': 448}, {'id': 720, 'image_count': 12}, {'id': 721, 'image_count': 18}, {'id': 722, 'image_count': 4}, {'id': 723, 'image_count': 71}, {'id': 724, 'image_count': 189}, {'id': 725, 'image_count': 74}, {'id': 726, 'image_count': 103}, {'id': 727, 'image_count': 3}, {'id': 728, 'image_count': 110}, {'id': 729, 'image_count': 5}, {'id': 730, 'image_count': 9}, {'id': 731, 'image_count': 15}, {'id': 732, 'image_count': 25}, {'id': 733, 'image_count': 7}, {'id': 734, 'image_count': 647}, {'id': 735, 'image_count': 824}, {'id': 736, 'image_count': 100}, {'id': 737, 'image_count': 47}, {'id': 738, 'image_count': 121}, {'id': 739, 'image_count': 731}, {'id': 740, 'image_count': 73}, {'id': 741, 'image_count': 49}, {'id': 742, 'image_count': 23}, {'id': 743, 'image_count': 4}, {'id': 744, 'image_count': 62}, {'id': 745, 'image_count': 118}, {'id': 746, 'image_count': 99}, {'id': 747, 'image_count': 40}, {'id': 748, 'image_count': 1036}, {'id': 749, 'image_count': 105}, {'id': 750, 'image_count': 21}, {'id': 751, 'image_count': 229}, {'id': 752, 'image_count': 7}, {'id': 753, 'image_count': 72}, {'id': 754, 'image_count': 9}, {'id': 755, 'image_count': 10}, {'id': 756, 'image_count': 328}, {'id': 757, 'image_count': 468}, {'id': 758, 'image_count': 1}, {'id': 759, 'image_count': 2}, {'id': 760, 'image_count': 24}, {'id': 761, 'image_count': 11}, {'id': 762, 'image_count': 72}, {'id': 763, 'image_count': 17}, {'id': 764, 'image_count': 10}, {'id': 765, 'image_count': 17}, {'id': 766, 'image_count': 489}, {'id': 767, 'image_count': 47}, {'id': 768, 'image_count': 93}, {'id': 769, 'image_count': 1}, {'id': 770, 'image_count': 12}, {'id': 771, 'image_count': 228}, {'id': 772, 'image_count': 5}, {'id': 773, 'image_count': 76}, {'id': 774, 'image_count': 71}, {'id': 775, 'image_count': 30}, {'id': 776, 'image_count': 109}, {'id': 777, 'image_count': 14}, {'id': 778, 'image_count': 1}, {'id': 779, 'image_count': 8}, {'id': 780, 'image_count': 26}, {'id': 781, 'image_count': 339}, {'id': 782, 'image_count': 153}, {'id': 783, 'image_count': 2}, {'id': 784, 'image_count': 3}, {'id': 785, 'image_count': 8}, {'id': 786, 'image_count': 47}, {'id': 787, 'image_count': 8}, {'id': 788, 'image_count': 6}, {'id': 789, 'image_count': 116}, {'id': 790, 'image_count': 69}, {'id': 791, 'image_count': 13}, {'id': 792, 'image_count': 6}, {'id': 793, 'image_count': 1928}, {'id': 794, 'image_count': 79}, {'id': 795, 'image_count': 14}, {'id': 796, 'image_count': 7}, {'id': 797, 'image_count': 20}, {'id': 798, 'image_count': 114}, {'id': 799, 'image_count': 221}, {'id': 800, 'image_count': 502}, {'id': 801, 'image_count': 62}, {'id': 802, 'image_count': 87}, {'id': 803, 'image_count': 4}, {'id': 804, 'image_count': 1912}, {'id': 805, 'image_count': 7}, {'id': 806, 'image_count': 186}, {'id': 807, 'image_count': 18}, {'id': 808, 'image_count': 4}, {'id': 809, 'image_count': 3}, {'id': 810, 'image_count': 7}, {'id': 811, 'image_count': 1413}, {'id': 812, 'image_count': 7}, {'id': 813, 'image_count': 12}, {'id': 814, 'image_count': 248}, {'id': 815, 'image_count': 4}, {'id': 816, 'image_count': 1881}, {'id': 817, 'image_count': 529}, {'id': 818, 'image_count': 1932}, {'id': 819, 'image_count': 50}, {'id': 820, 'image_count': 3}, {'id': 821, 'image_count': 28}, {'id': 822, 'image_count': 10}, {'id': 823, 'image_count': 5}, {'id': 824, 'image_count': 5}, {'id': 825, 'image_count': 18}, {'id': 826, 'image_count': 14}, {'id': 827, 'image_count': 1890}, {'id': 828, 'image_count': 660}, {'id': 829, 'image_count': 8}, {'id': 830, 'image_count': 25}, {'id': 831, 'image_count': 10}, {'id': 832, 'image_count': 218}, {'id': 833, 'image_count': 36}, {'id': 834, 'image_count': 16}, {'id': 835, 'image_count': 808}, {'id': 836, 'image_count': 479}, {'id': 837, 'image_count': 1404}, {'id': 838, 'image_count': 307}, {'id': 839, 'image_count': 57}, {'id': 840, 'image_count': 28}, {'id': 841, 'image_count': 80}, {'id': 842, 'image_count': 11}, {'id': 843, 'image_count': 92}, {'id': 844, 'image_count': 20}, {'id': 845, 'image_count': 194}, {'id': 846, 'image_count': 23}, {'id': 847, 'image_count': 52}, {'id': 848, 'image_count': 673}, {'id': 849, 'image_count': 2}, {'id': 850, 'image_count': 2}, {'id': 851, 'image_count': 1}, {'id': 852, 'image_count': 2}, {'id': 853, 'image_count': 8}, {'id': 854, 'image_count': 80}, {'id': 855, 'image_count': 3}, {'id': 856, 'image_count': 3}, {'id': 857, 'image_count': 15}, {'id': 858, 'image_count': 2}, {'id': 859, 'image_count': 10}, {'id': 860, 'image_count': 386}, {'id': 861, 'image_count': 65}, {'id': 862, 'image_count': 3}, {'id': 863, 'image_count': 35}, {'id': 864, 'image_count': 5}, {'id': 865, 'image_count': 180}, {'id': 866, 'image_count': 99}, {'id': 867, 'image_count': 49}, {'id': 868, 'image_count': 28}, {'id': 869, 'image_count': 1}, {'id': 870, 'image_count': 52}, {'id': 871, 'image_count': 36}, {'id': 872, 'image_count': 70}, {'id': 873, 'image_count': 6}, {'id': 874, 'image_count': 29}, {'id': 875, 'image_count': 24}, {'id': 876, 'image_count': 1115}, {'id': 877, 'image_count': 61}, {'id': 878, 'image_count': 18}, {'id': 879, 'image_count': 18}, {'id': 880, 'image_count': 665}, {'id': 881, 'image_count': 1096}, {'id': 882, 'image_count': 29}, {'id': 883, 'image_count': 8}, {'id': 884, 'image_count': 14}, {'id': 885, 'image_count': 1622}, {'id': 886, 'image_count': 2}, {'id': 887, 'image_count': 3}, {'id': 888, 'image_count': 32}, {'id': 889, 'image_count': 55}, {'id': 890, 'image_count': 1}, {'id': 891, 'image_count': 10}, {'id': 892, 'image_count': 10}, {'id': 893, 'image_count': 47}, {'id': 894, 'image_count': 3}, {'id': 895, 'image_count': 29}, {'id': 896, 'image_count': 342}, {'id': 897, 'image_count': 25}, {'id': 898, 'image_count': 1469}, {'id': 899, 'image_count': 521}, {'id': 900, 'image_count': 347}, {'id': 901, 'image_count': 35}, {'id': 902, 'image_count': 7}, {'id': 903, 'image_count': 207}, {'id': 904, 'image_count': 108}, {'id': 905, 'image_count': 2}, {'id': 906, 'image_count': 34}, {'id': 907, 'image_count': 12}, {'id': 908, 'image_count': 10}, {'id': 909, 'image_count': 13}, {'id': 910, 'image_count': 361}, {'id': 911, 'image_count': 1023}, {'id': 912, 'image_count': 782}, {'id': 913, 'image_count': 2}, {'id': 914, 'image_count': 5}, {'id': 915, 'image_count': 247}, {'id': 916, 'image_count': 221}, {'id': 917, 'image_count': 4}, {'id': 918, 'image_count': 8}, {'id': 919, 'image_count': 158}, {'id': 920, 'image_count': 3}, {'id': 921, 'image_count': 752}, {'id': 922, 'image_count': 64}, {'id': 923, 'image_count': 707}, {'id': 924, 'image_count': 143}, {'id': 925, 'image_count': 1}, {'id': 926, 'image_count': 49}, {'id': 927, 'image_count': 126}, {'id': 928, 'image_count': 76}, {'id': 929, 'image_count': 11}, {'id': 930, 'image_count': 11}, {'id': 931, 'image_count': 4}, {'id': 932, 'image_count': 39}, {'id': 933, 'image_count': 11}, {'id': 934, 'image_count': 13}, {'id': 935, 'image_count': 91}, {'id': 936, 'image_count': 14}, {'id': 937, 'image_count': 5}, {'id': 938, 'image_count': 3}, {'id': 939, 'image_count': 10}, {'id': 940, 'image_count': 18}, {'id': 941, 'image_count': 9}, {'id': 942, 'image_count': 6}, {'id': 943, 'image_count': 951}, {'id': 944, 'image_count': 2}, {'id': 945, 'image_count': 1}, {'id': 946, 'image_count': 19}, {'id': 947, 'image_count': 1942}, {'id': 948, 'image_count': 1916}, {'id': 949, 'image_count': 139}, {'id': 950, 'image_count': 43}, {'id': 951, 'image_count': 1969}, {'id': 952, 'image_count': 5}, {'id': 953, 'image_count': 134}, {'id': 954, 'image_count': 74}, {'id': 955, 'image_count': 381}, {'id': 956, 'image_count': 1}, {'id': 957, 'image_count': 381}, {'id': 958, 'image_count': 6}, {'id': 959, 'image_count': 1826}, {'id': 960, 'image_count': 28}, {'id': 961, 'image_count': 1635}, {'id': 962, 'image_count': 1967}, {'id': 963, 'image_count': 16}, {'id': 964, 'image_count': 1926}, {'id': 965, 'image_count': 1789}, {'id': 966, 'image_count': 401}, {'id': 967, 'image_count': 1968}, {'id': 968, 'image_count': 1167}, {'id': 969, 'image_count': 1}, {'id': 970, 'image_count': 56}, {'id': 971, 'image_count': 17}, {'id': 972, 'image_count': 1}, {'id': 973, 'image_count': 58}, {'id': 974, 'image_count': 9}, {'id': 975, 'image_count': 8}, {'id': 976, 'image_count': 1124}, {'id': 977, 'image_count': 31}, {'id': 978, 'image_count': 16}, {'id': 979, 'image_count': 491}, {'id': 980, 'image_count': 432}, {'id': 981, 'image_count': 1945}, {'id': 982, 'image_count': 1899}, {'id': 983, 'image_count': 5}, {'id': 984, 'image_count': 28}, {'id': 985, 'image_count': 7}, {'id': 986, 'image_count': 146}, {'id': 987, 'image_count': 1}, {'id': 988, 'image_count': 25}, {'id': 989, 'image_count': 22}, {'id': 990, 'image_count': 1}, {'id': 991, 'image_count': 10}, {'id': 992, 'image_count': 9}, {'id': 993, 'image_count': 308}, {'id': 994, 'image_count': 4}, {'id': 995, 'image_count': 1969}, {'id': 996, 'image_count': 45}, {'id': 997, 'image_count': 12}, {'id': 998, 'image_count': 1}, {'id': 999, 'image_count': 85}, {'id': 1000, 'image_count': 1127}, {'id': 1001, 'image_count': 11}, {'id': 1002, 'image_count': 60}, {'id': 1003, 'image_count': 1}, {'id': 1004, 'image_count': 16}, {'id': 1005, 'image_count': 1}, {'id': 1006, 'image_count': 65}, {'id': 1007, 'image_count': 13}, {'id': 1008, 'image_count': 655}, {'id': 1009, 'image_count': 51}, {'id': 1010, 'image_count': 1}, {'id': 1011, 'image_count': 673}, {'id': 1012, 'image_count': 5}, {'id': 1013, 'image_count': 36}, {'id': 1014, 'image_count': 54}, {'id': 1015, 'image_count': 5}, {'id': 1016, 'image_count': 8}, {'id': 1017, 'image_count': 305}, {'id': 1018, 'image_count': 297}, {'id': 1019, 'image_count': 1053}, {'id': 1020, 'image_count': 223}, {'id': 1021, 'image_count': 1037}, {'id': 1022, 'image_count': 63}, {'id': 1023, 'image_count': 1881}, {'id': 1024, 'image_count': 507}, {'id': 1025, 'image_count': 333}, {'id': 1026, 'image_count': 1911}, {'id': 1027, 'image_count': 1765}, {'id': 1028, 'image_count': 1}, {'id': 1029, 'image_count': 5}, {'id': 1030, 'image_count': 1}, {'id': 1031, 'image_count': 9}, {'id': 1032, 'image_count': 2}, {'id': 1033, 'image_count': 151}, {'id': 1034, 'image_count': 82}, {'id': 1035, 'image_count': 1931}, {'id': 1036, 'image_count': 41}, {'id': 1037, 'image_count': 1895}, {'id': 1038, 'image_count': 24}, {'id': 1039, 'image_count': 22}, {'id': 1040, 'image_count': 35}, {'id': 1041, 'image_count': 69}, {'id': 1042, 'image_count': 962}, {'id': 1043, 'image_count': 588}, {'id': 1044, 'image_count': 21}, {'id': 1045, 'image_count': 825}, {'id': 1046, 'image_count': 52}, {'id': 1047, 'image_count': 5}, {'id': 1048, 'image_count': 5}, {'id': 1049, 'image_count': 5}, {'id': 1050, 'image_count': 1860}, {'id': 1051, 'image_count': 56}, {'id': 1052, 'image_count': 1582}, {'id': 1053, 'image_count': 7}, {'id': 1054, 'image_count': 2}, {'id': 1055, 'image_count': 1562}, {'id': 1056, 'image_count': 1885}, {'id': 1057, 'image_count': 1}, {'id': 1058, 'image_count': 5}, {'id': 1059, 'image_count': 137}, {'id': 1060, 'image_count': 1094}, {'id': 1061, 'image_count': 134}, {'id': 1062, 'image_count': 29}, {'id': 1063, 'image_count': 22}, {'id': 1064, 'image_count': 522}, {'id': 1065, 'image_count': 50}, {'id': 1066, 'image_count': 68}, {'id': 1067, 'image_count': 16}, {'id': 1068, 'image_count': 40}, {'id': 1069, 'image_count': 35}, {'id': 1070, 'image_count': 135}, {'id': 1071, 'image_count': 1413}, {'id': 1072, 'image_count': 772}, {'id': 1073, 'image_count': 50}, {'id': 1074, 'image_count': 1015}, {'id': 1075, 'image_count': 1}, {'id': 1076, 'image_count': 65}, {'id': 1077, 'image_count': 1900}, {'id': 1078, 'image_count': 1302}, {'id': 1079, 'image_count': 1977}, {'id': 1080, 'image_count': 2}, {'id': 1081, 'image_count': 29}, {'id': 1082, 'image_count': 36}, {'id': 1083, 'image_count': 138}, {'id': 1084, 'image_count': 4}, {'id': 1085, 'image_count': 67}, {'id': 1086, 'image_count': 26}, {'id': 1087, 'image_count': 25}, {'id': 1088, 'image_count': 33}, {'id': 1089, 'image_count': 37}, {'id': 1090, 'image_count': 50}, {'id': 1091, 'image_count': 270}, {'id': 1092, 'image_count': 12}, {'id': 1093, 'image_count': 316}, {'id': 1094, 'image_count': 41}, {'id': 1095, 'image_count': 224}, {'id': 1096, 'image_count': 105}, {'id': 1097, 'image_count': 1925}, {'id': 1098, 'image_count': 1021}, {'id': 1099, 'image_count': 1213}, {'id': 1100, 'image_count': 172}, {'id': 1101, 'image_count': 28}, {'id': 1102, 'image_count': 745}, {'id': 1103, 'image_count': 187}, {'id': 1104, 'image_count': 147}, {'id': 1105, 'image_count': 136}, {'id': 1106, 'image_count': 34}, {'id': 1107, 'image_count': 41}, {'id': 1108, 'image_count': 636}, {'id': 1109, 'image_count': 570}, {'id': 1110, 'image_count': 1149}, {'id': 1111, 'image_count': 61}, {'id': 1112, 'image_count': 1890}, {'id': 1113, 'image_count': 18}, {'id': 1114, 'image_count': 143}, {'id': 1115, 'image_count': 1517}, {'id': 1116, 'image_count': 7}, {'id': 1117, 'image_count': 943}, {'id': 1118, 'image_count': 6}, {'id': 1119, 'image_count': 1}, {'id': 1120, 'image_count': 11}, {'id': 1121, 'image_count': 101}, {'id': 1122, 'image_count': 1909}, {'id': 1123, 'image_count': 800}, {'id': 1124, 'image_count': 1}, {'id': 1125, 'image_count': 44}, {'id': 1126, 'image_count': 3}, {'id': 1127, 'image_count': 44}, {'id': 1128, 'image_count': 31}, {'id': 1129, 'image_count': 7}, {'id': 1130, 'image_count': 20}, {'id': 1131, 'image_count': 11}, {'id': 1132, 'image_count': 13}, {'id': 1133, 'image_count': 1924}, {'id': 1134, 'image_count': 113}, {'id': 1135, 'image_count': 2}, {'id': 1136, 'image_count': 139}, {'id': 1137, 'image_count': 12}, {'id': 1138, 'image_count': 37}, {'id': 1139, 'image_count': 1866}, {'id': 1140, 'image_count': 47}, {'id': 1141, 'image_count': 1468}, {'id': 1142, 'image_count': 729}, {'id': 1143, 'image_count': 24}, {'id': 1144, 'image_count': 1}, {'id': 1145, 'image_count': 10}, {'id': 1146, 'image_count': 3}, {'id': 1147, 'image_count': 14}, {'id': 1148, 'image_count': 4}, {'id': 1149, 'image_count': 29}, {'id': 1150, 'image_count': 4}, {'id': 1151, 'image_count': 70}, {'id': 1152, 'image_count': 46}, {'id': 1153, 'image_count': 14}, {'id': 1154, 'image_count': 48}, {'id': 1155, 'image_count': 1855}, {'id': 1156, 'image_count': 113}, {'id': 1157, 'image_count': 1}, {'id': 1158, 'image_count': 1}, {'id': 1159, 'image_count': 10}, {'id': 1160, 'image_count': 54}, {'id': 1161, 'image_count': 1923}, {'id': 1162, 'image_count': 630}, {'id': 1163, 'image_count': 31}, {'id': 1164, 'image_count': 69}, {'id': 1165, 'image_count': 7}, {'id': 1166, 'image_count': 11}, {'id': 1167, 'image_count': 1}, {'id': 1168, 'image_count': 30}, {'id': 1169, 'image_count': 50}, {'id': 1170, 'image_count': 45}, {'id': 1171, 'image_count': 28}, {'id': 1172, 'image_count': 114}, {'id': 1173, 'image_count': 193}, {'id': 1174, 'image_count': 21}, {'id': 1175, 'image_count': 91}, {'id': 1176, 'image_count': 31}, {'id': 1177, 'image_count': 1469}, {'id': 1178, 'image_count': 1924}, {'id': 1179, 'image_count': 87}, {'id': 1180, 'image_count': 77}, {'id': 1181, 'image_count': 11}, {'id': 1182, 'image_count': 47}, {'id': 1183, 'image_count': 21}, {'id': 1184, 'image_count': 47}, {'id': 1185, 'image_count': 70}, {'id': 1186, 'image_count': 1838}, {'id': 1187, 'image_count': 19}, {'id': 1188, 'image_count': 531}, {'id': 1189, 'image_count': 11}, {'id': 1190, 'image_count': 941}, {'id': 1191, 'image_count': 113}, {'id': 1192, 'image_count': 26}, {'id': 1193, 'image_count': 5}, {'id': 1194, 'image_count': 56}, {'id': 1195, 'image_count': 73}, {'id': 1196, 'image_count': 32}, {'id': 1197, 'image_count': 128}, {'id': 1198, 'image_count': 623}, {'id': 1199, 'image_count': 12}, {'id': 1200, 'image_count': 52}, {'id': 1201, 'image_count': 11}, {'id': 1202, 'image_count': 1674}, {'id': 1203, 'image_count': 81}]  # noqa
# fmt: on


def get_fed_loss_cls_weights(freq_weight_power=0.5):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """

    class_freq_meta = LVIS_CATEGORY_IMAGE_COUNT
    class_freq = torch.tensor(
        [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
    )
    class_freq_weight = class_freq.float() ** freq_weight_power
    return class_freq_weight



# copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
def get_fed_loss_classes(gt_classes, num_fed_loss_classes, num_classes, weight):
    """
    Args:
        gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
        Will sample negative classes if number of unique gt_classes is smaller than this value.
        num_classes: number of foreground classes
        weight: probabilities used to sample negative classes
    Returns:
        Tensor:
            classes to keep when calculating the federated loss, including both unique gt
            classes and sampled negative classes.
    """
    unique_gt_classes = torch.unique(gt_classes)
    prob = unique_gt_classes.new_ones(num_classes + 1).float()
    prob[-1] = 0
    if len(unique_gt_classes) < num_fed_loss_classes:
        prob[:num_classes] = weight.float().clone()
        prob[unique_gt_classes] = 0
        sampled_negative_classes = torch.multinomial(
            prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
        )
        fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
    else:
        fed_loss_classes = unique_gt_classes
    return fed_loss_classes

# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          fed_loss_num_classes,
                          fed_loss_cls_weights,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    
    fed_loss_classes = get_fed_loss_classes(
        target,
        num_fed_loss_classes=fed_loss_num_classes,
        num_classes=1203,
        weight=fed_loss_cls_weights.to(target.device),
    )
    fed_loss_classes_mask = fed_loss_classes.new_zeros(1204)
    fed_loss_classes_mask[fed_loss_classes] = 1
    fed_loss_classes_mask = fed_loss_classes_mask[:1203]
    fed_weight = fed_loss_classes_mask.view(1, 1203).float()
    loss = loss * fed_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       fed_loss_num_classes,
                       fed_loss_cls_weights,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    
    fed_loss_classes = get_fed_loss_classes(
        target,
        num_fed_loss_classes=fed_loss_num_classes,
        num_classes=1203,
        weight=fed_loss_cls_weights.to(target.device),
    )
    fed_loss_classes_mask = fed_loss_classes.new_zeros(1204)
    fed_loss_classes_mask[fed_loss_classes] = 1
    fed_loss_classes_mask = fed_loss_classes_mask[:1203]
    fed_weight = fed_loss_classes_mask.view(1, 1203).float()
    loss = loss * fed_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def quality_focal_loss(pred, target, fed_loss_num_classes,
                          fed_loss_cls_weights, beta=2.0, ):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    fed_loss_classes = get_fed_loss_classes(
        label,
        num_fed_loss_classes=fed_loss_num_classes,
        num_classes=1203,
        weight=fed_loss_cls_weights.to(label.device),
    )
    fed_loss_classes_mask = fed_loss_classes.new_zeros(1204)
    fed_loss_classes_mask[fed_loss_classes] = 1
    fed_loss_classes_mask = fed_loss_classes_mask[:1203]
    fed_weight = fed_loss_classes_mask.view(1, 1203).float()
    loss = loss * fed_weight

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def quality_focal_loss_with_prob(pred, target, fed_loss_num_classes,
                          fed_loss_cls_weights, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Different from `quality_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)
    
    fed_loss_classes = get_fed_loss_classes(
        label,
        num_fed_loss_classes=fed_loss_num_classes,
        num_classes=1203,
        weight=fed_loss_cls_weights.to(label.device),
    )
    fed_loss_classes_mask = fed_loss_classes.new_zeros(1204)
    fed_loss_classes_mask[fed_loss_classes] = 1
    fed_loss_classes_mask = fed_loss_classes_mask[:1203]
    fed_weight = fed_loss_classes_mask.view(1, 1203).float()
    loss = loss * fed_weight

    loss = loss.sum(dim=1, keepdim=False)
    return loss



@LOSSES.register_module()
class FederatedLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 fed_loss_num_classes=50):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.\
        """
        super(FederatedLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.fed_loss_num_classes = fed_loss_num_classes
        self.fed_loss_cls_weights = get_fed_loss_cls_weights()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                self.fed_loss_num_classes,
                self.fed_loss_cls_weights,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls




@LOSSES.register_module()
class FederatedQualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,
                 fed_loss_num_classes=50,
                 open_vocabulary=False,
                 base_ind_file=None):
        super(FederatedQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.fed_loss_num_classes = fed_loss_num_classes
        self.fed_loss_cls_weights = get_fed_loss_cls_weights()
        if open_vocabulary:
            base_inds = open(base_ind_file, 'r').readline().strip().split(', ')
            base_inds = [int(ind) for ind in base_inds]
            novel_inds = [i for i in range(1203) if i not in base_inds]
            novel_inds_tensor = torch.tensor(novel_inds, device=self.fed_loss_cls_weights.device)
            self.fed_loss_cls_weights[novel_inds_tensor] = 0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                calculate_loss_func = quality_focal_loss
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                fed_loss_num_classes=self.fed_loss_num_classes,
                fed_loss_cls_weights=self.fed_loss_cls_weights,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls




