import backbone
import torch

if __name__ == "__main__":
    encoder = backbone.EncoderNet()
    decoder = backbone.DecoderNet()

    test_data = torch.ones(1, 3, 260, 392)

    test_out1 = encoder(test_data)
    test_out2 = decoder(test_out1)
    print('done')