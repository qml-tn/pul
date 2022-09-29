import torch
import numpy as np


def linear_encoder(input):
    emb_list = [input, 1-input]
    return torch.stack(emb_list, dim=2)


def linear_decoder(input):
    # x = input/torch.linalg.norm(input, axis=-1, keepdims=True)
    x = torch.abs(input)
    x = x/(torch.sum(x, axis=-1, keepdims=True)+1e-6)
    return x[:, :, 0]


def angle_encoder(input):
    pihalf = np.pi/2.0
    x = torch.stack(
        [torch.cos(input*pihalf), torch.sin(input*pihalf)], axis=-1)
    return x


def angle_decoder(input):
    pihalf = np.pi/2.0
    x = input[..., 0]
    y = input[..., 1]
    x = torch.atan2(y, x)
    # x.shape  (nbatch,N)
    return x/pihalf


def embedding2D(input, d, shift=None):
    """ Function to embed the tensor of size [nbatch, n] into a tensor of [nbatch, n, d]
        We use an extension of the embedding which is used in the library for the case of d=2;
        input:  the input vector
        d:      local hilbert space dimension
        shift:  shift used in the embedding in order to combine more points of
                the vector in a local embedding
    """
    n = input.shape[1]
    if shift is None:
        shift = int(np.sqrt(n))

    input_list = [input]
    for i in range(d-1):
        input_list.append(1-left_cyclic_shift(input, i*shift))
    embedded_data = torch.stack(input_list, dim=2)
    return embedded_data


def image_embedding(input, embedding="linear", aug_phi=1e-3, embedding_order=1):
    """ Function to embed an image tensor of size [nbatch, nchan, size_x, size_y] into a tensor of [nbatch, n, nchan]
        input:      the input vector
        embedding:  Embedding type: linear (default), angle
        aug_phi:    Random shift of the input vector elements.
        output:     torch tensor of size [batch_size, input_dim, feature_dim]
    """
    shape = input.shape
    nbatch = shape[0]
    nchan = shape[1]
    n = np.prod(shape[2:])

    x = torch.reshape(input, (nbatch, nchan, -1))/nchan

    x += torch.rand(*x.shape)*aug_phi/nchan

    x = torch.clip(x, 0.0, 1.0)

    if embedding == "linear":
        if embedding_order == 1:
            x = torch.cat(
                [1 - torch.sum(x, dim=1, keepdim=True), x], dim=1)
        elif embedding_order == 2:
            x2 = torch.reshape(torch.einsum(
                "bin,bjn->bijn", x, x), [nbatch, nchan**2, n])
            x = torch.cat(
                [1 - torch.sum(x, dim=1, keepdim=True), x, x2], dim=1)
            # We normalize to L1 probability
            x = x/torch.sum(x, dim=1, keepdim=True)
        elif embedding_order == 3:
            x2 = torch.reshape(torch.einsum(
                "bin,bjn->bijn", x, x), [nbatch, nchan**2, n])
            x3 = torch.reshape(torch.einsum(
                "bin,bjn,bkn->bijkn", x, x, x), [nbatch, nchan**3, n])
            x = torch.cat(
                [1 - torch.sum(x, dim=1, keepdim=True), x, x2, x3], dim=1)
            # We normalize to L1 probability
            x = x/torch.sum(x, dim=1, keepdim=True)
        else:
            raise Exception(
                f"Implemented embedding orders are: [1, 2, 3] but got {embedding_order}")
    else:
        pi_half = np.pi/2.
        cx = torch.cos(x*pi_half)
        sx = torch.sin(x*pi_half)
        xlist = [torch.prod(cx, dim=1, keepdim=True)]
        for i in range(1, nchan):
            x1 = torch.prod(cx[i:], dim=1, keepdim=True)
            x2 = torch.prod(sx[:i], dim=1, keepdim=True)
            xlist.append(x1*x2)
        xlist.append(torch.prod(sx, dim=1, keepdim=True))
        x = torch.cat(xlist, dim=1)

    return torch.transpose(x, 1, 2)


spiral25 = [[0, 1, 2, 3, 4, 29, 30, 31, 32, 49, 50, 51, 52, 53, 54, 79, 80, 81,
             82, 99, 100, 101, 102, 103, 104], [17, 16, 15, 14, 5, 28, 39, 38,
                                                33, 48, 67, 66, 65, 64, 55, 78, 89, 88, 83, 98, 117, 116, 115, 114,
                                                105], [18, 11, 12, 13, 6, 27, 40, 37, 34, 47, 68, 61, 62, 63, 56,
                                                       77, 90, 87, 84, 97, 118, 111, 112, 113, 106], [19, 10, 9, 8, 7, 26,
                                                                                                      41, 36, 35, 46, 69, 60, 59, 58, 57, 76, 91, 86, 85, 96, 119, 110,
                                                                                                      109, 108, 107], [20, 21, 22, 23, 24, 25, 42, 43, 44, 45, 70, 71, 72,
                                                                                                                       73, 74, 75, 92, 93, 94, 95, 120, 121, 122, 123, 124], [445, 444,
                                                                                                                                                                              443, 442, 425, 424, 423, 422, 421, 420, 395, 394, 393, 392, 375,
                                                                                                                                                                              374, 373, 372, 371, 370, 145, 144, 143, 142, 125], [446, 435, 436,
                                                                                                                                                                                                                                  441, 426, 407, 408, 409, 410, 419, 396, 385, 386, 391, 376, 357,
                                                                                                                                                                                                                                  358, 359, 360, 369, 146, 135, 136, 141, 126], [447, 434, 437, 440,
                                                                                                                                                                                                                                                                                 427, 406, 413, 412, 411, 418, 397, 384, 387, 390, 377, 356, 363,
                                                                                                                                                                                                                                                                                 362, 361, 368, 147, 134, 137, 140, 127], [448, 433, 438, 439, 428,
                                                                                                                                                                                                                                                                                                                           405, 414, 415, 416, 417, 398, 383, 388, 389, 378, 355, 364, 365,
                                                                                                                                                                                                                                                                                                                           366, 367, 148, 133, 138, 139, 128], [449, 432, 431, 430, 429, 404,
                                                                                                                                                                                                                                                                                                                                                                403, 402, 401, 400, 399, 382, 381, 380, 379, 354, 353, 352, 351,
                                                                                                                                                                                                                                                                                                                                                                350, 149, 132, 131, 130, 129], [450, 451, 452, 453, 454, 279, 280,
                                                                                                                                                                                                                                                                                                                                                                                                281, 282, 299, 300, 301, 302, 303, 304, 329, 330, 331, 332, 349,
                                                                                                                                                                                                                                                                                                                                                                                                150, 151, 152, 153, 154], [467, 466, 465, 464, 455, 278, 289, 288,
                                                                                                                                                                                                                                                                                                                                                                                                                           283, 298, 317, 316, 315, 314, 305, 328, 339, 338, 333, 348, 167,
                                                                                                                                                                                                                                                                                                                                                                                                                           166, 165, 164, 155], [468, 461, 462, 463, 456, 277, 290, 287, 284,
                                                                                                                                                                                                                                                                                                                                                                                                                                                 297, 318, 311, 312, 313, 306, 327, 340, 337, 334, 347, 168, 161,
                                                                                                                                                                                                                                                                                                                                                                                                                                                 162, 163, 156], [469, 460, 459, 458, 457, 276, 291, 286, 285, 296,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                  319, 310, 309, 308, 307, 326, 341, 336, 335, 346, 169, 160, 159,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                  158, 157], [470, 471, 472, 473, 474, 275, 292, 293, 294, 295, 320,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              321, 322, 323, 324, 325, 342, 343, 344, 345, 170, 171, 172, 173,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              174], [495, 494, 493, 492, 475, 274, 273, 272, 271, 270, 245, 244,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     243, 242, 225, 224, 223, 222, 221, 220, 195, 194, 193, 192,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     175], [496, 485, 486, 491, 476, 257, 258, 259, 260, 269, 246, 235,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            236, 241, 226, 207, 208, 209, 210, 219, 196, 185, 186, 191,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            176], [497, 484, 487, 490, 477, 256, 263, 262, 261, 268, 247, 234,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   237, 240, 227, 206, 213, 212, 211, 218, 197, 184, 187, 190,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   177], [498, 483, 488, 489, 478, 255, 264, 265, 266, 267, 248, 233,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          238, 239, 228, 205, 214, 215, 216, 217, 198, 183, 188, 189,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          178], [499, 482, 481, 480, 479, 254, 253, 252, 251, 250, 249, 232,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 231, 230, 229, 204, 203, 202, 201, 200, 199, 182, 181, 180,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 179], [500, 501, 502, 503, 504, 529, 530, 531, 532, 549, 550, 551,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        552, 553, 554, 579, 580, 581, 582, 599, 600, 601, 602, 603,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        604], [517, 516, 515, 514, 505, 528, 539, 538, 533, 548, 567, 566,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               565, 564, 555, 578, 589, 588, 583, 598, 617, 616, 615, 614,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               605], [518, 511, 512, 513, 506, 527, 540, 537, 534, 547, 568, 561,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      562, 563, 556, 577, 590, 587, 584, 597, 618, 611, 612, 613,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      606], [519, 510, 509, 508, 507, 526, 541, 536, 535, 546, 569, 560,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             559, 558, 557, 576, 591, 586, 585, 596, 619, 610, 609, 608,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             607], [520, 521, 522, 523, 524, 525, 542, 543, 544, 545, 570, 571,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    572, 573, 574, 575, 592, 593, 594, 595, 620, 621, 622, 623, 624]]


def spiral_inds(row):
    if row == 25:
        return list(np.reshape(spiral25, [-1]))
    lis = [[0 for i in range(0, row)] for j in range(0, row)]
    s = []
    if row > 1:
        s += [row-1]
        for i in range(row-1, 0, -1):
            s += [i, i]
    b = 1
    e = 0
    a = 0
    c = 0
    d = 0
    lis[0][0] = e
    for n in s:
        for f in range(n):
            c += a
            d += b
            e += 1
            lis[c][d] = e
        a, b = b, -a
    return list(np.reshape(lis, [-1]))


def left_cyclic_shift(input, shift):
    if shift == 0:
        return input
    n = input.shape[1]
    inds = list(range(n))
    inds = inds[shift:]+inds[:shift]
    shifted_input = input[:, inds]
    return shifted_input
