import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.networks import define_G
import glob
import gc
import torch.cuda
import warnings
import cv2
import os
from pathlib import Path


# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def rescale(image, Rescale=True):
    if not Rescale:
        return image
    if Rescale:
        width, height = image.size
        threshold = 1024
        # calculate scaling factor to make the larger dimension threshold
        scale = min(threshold / width, threshold / height)
        if scale < 1:  # resize if image is larger than threshold in any dimension
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.BICUBIC)
        #handle images smaller than 128 pixels
        width, height = image.size
        while width < 128 or height < 128:
            image = image.resize((int(width * 2), int(height * 2)), Image.BICUBIC)
            width, height = image.size
        return image


# Extract MLP code in advance
MLP_code = [
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156,
    -31543.1289, 50374.1289, 99631.4062, -188897.3750, 138322.7031,
    -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.6250,
    -69972.6875, -21886.4141, 86938.4766, 31457.6270, -98892.2344,
    -1191.5887, -61662.1719, -180121.9062, -32931.0859, 43109.0391,
    21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125,
    52496.7422, 142697.4062, -34882.7852, -98740.0625, 34458.5078,
    -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562,
    69853.3906, 79046.2656, -3923.7344, -5644.5229, 96586.7578,
    -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703,
    -75055.0469, 19571.8203, 79358.7891, -16501.5000, -147169.2188,
    -97861.6797, 60442.1797, 40156.9023, 223136.3906, -81118.0547,
    -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844,
    40865.9609, -28627.9043, -18604.7227, 120274.6172, 49712.2383,
    164402.7031, -53165.0820, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578,
    -143190.0625, -74049.5312, 137980.5781, -150788.7656, -115719.6719,
    -189250.1250, -153069.7344, -127429.7891, -187588.2500, 125264.7422,
    -79082.3438, -114144.5781, 36033.5039, -57502.2188, 80488.1562,
    36501.4570, -138817.5938, -22189.6523, -222146.9688, -73292.3984,
    127717.2422, -183836.3750, -105907.0859, 145422.8750, 66981.2031,
    -9596.6699, 78099.4922, 70226.3359, 35841.8789, -116117.6016,
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375,
    -32068.8652, 160663.3438, 72187.0703, -148519.5469, 162952.8906,
    -128048.2031, -136153.8906, -15270.3730, -52766.3281, -52517.4531,
    18652.1992, 195354.2188, -136657.3750, -8034.2622, -92699.6016,
    -129169.1406, 188479.9844, 46003.7500, -93383.0781, -67831.6484,
    -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203,
    71213.1250, 94603.1094, -30359.8125, -107989.2578, 99822.1719,
    184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.7930, 50421.4141, 60784.4102, -18286.3398,
    -182851.9531, -87178.7969, -13131.7539, 195674.8906, 59951.7852,
    124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102,
    -64394.3438, 113281.1797, -93987.0703, 221989.7188, 132902.5000,
    -9538.8574, -14594.1338, 65084.9453, -12501.7227, 130330.6875,
    -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656,
    -186678.8281, -166799.9375, 138770.6250, -78969.9531, 124516.8047,
    -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529,
    132985.8281, 163929.8438, -188517.7969, -141155.6406, 45071.0391,
    207788.3125, -125826.1172, 8965.3320, -159584.8438, 95842.4609,
    -76929.4688
]

class Model():
    def __init__(self,model_name, device="cuda"):
        self.device = torch.device(device)
        self.G_A_net = None
        self.alias_net = None
        self.ref_t = None
        self.cell_size_code = None
        self.model_name = model_name
        # Load anime model from torch hub
        self.anime_model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2").to(device)
        self.anime_model.eval()
        
    def load(self):
        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])
            G_A_state = torch.load("./checkpoints/{}/160_net_G_A.pth".format(self.model_name), 
                                 map_location=str(self.device),
                                 weights_only=True)
            for p in list(G_A_state.keys()):
                G_A_state["module."+str(p)] = G_A_state.pop(p)
            self.G_A_net.load_state_dict(G_A_state)
            alias_state = torch.load("./alias_net.pth", 
                                   map_location=str(self.device),
                                   weights_only=True)
            for p in list(alias_state.keys()):
                alias_state["module."+str(p)] = alias_state.pop(p)
            self.alias_net.load_state_dict(alias_state)
            code = torch.tensor(MLP_code, device=self.device).reshape((1, 256, 1, 1))
            self.cell_size_code = self.G_A_net.module.MLP(code)
            self.G_A_net.eval()
            self.alias_net.eval()

    def animeize_image(self, image_path):
        """Convert a real image to anime style using torch hub model"""
        # Read and convert image
        img = Image.open(image_path).convert('RGB')
        
        # Transform image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Generate anime style
        with torch.no_grad():
            output = self.anime_model(img_tensor)
        
        # Convert back to image
        output = output.squeeze(0).cpu()
        output = (output + 1) / 2.0 * 255.0
        output = output.clamp(0, 255).to(torch.uint8)
        output = output.permute(1, 2, 0).numpy()
        output_img = Image.fromarray(output)
        
        # Save intermediate result
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_anime{ext}"
        output_img.save(output_path)
        return output_path

    def pixelize(self, in_img, out_img, cell_size):
        with torch.no_grad():
            in_img = Image.open(in_img).convert('RGB')
            width, height = in_img.size
            best_cell_size = 4
            
            # Process the entire image at once
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img_tensor = trans(in_img).unsqueeze(0).to(self.device)
            
            # Generate pixelized image
            feature = self.G_A_net.module.RGBEnc(img_tensor)
            images = self.G_A_net.module.RGBDec(feature, self.cell_size_code)
            out_t = self.alias_net(images)
            
            # Convert tensor to image
            out_img_tensor = out_t[0].cpu().float()
            out_img_tensor = (out_img_tensor + 1) / 2.0 * 255.0
            out_img_tensor = out_img_tensor.clamp(0, 255).to(torch.uint8)
            out_img_tensor = out_img_tensor.permute(1, 2, 0).numpy()
            merged_img = Image.fromarray(out_img_tensor)
            
            # Resize to final dimensions
            merged_img = merged_img.resize(((width // cell_size) * best_cell_size, 
                                         (height // cell_size) * best_cell_size), 
                                         Image.NEAREST)
            merged_img = merged_img.resize((width, height), Image.NEAREST)
            merged_img.save(out_img)

def load_binary_classifier():
    """Load the binary classifier model"""
    model = torch.load('binaryClassifier/realunreal_classifier.pt')
    return model

def is_real_image(image_path, classifier):
    """Determine if an image is real using the binary classifier"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = classifier['model'](img_tensor)
        _, predicted = torch.max(output, 1)
        
    return predicted.item() == 0  # 0 for real, 1 for unreal

def pixelize_cli():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Pixelization')
    parser.add_argument('--input', type=str, default=None, required=True, help='path to image or directory')
    parser.add_argument('--output', type=str, default=None, required=False, help='path to save image/images')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--cell_size', type=int, default=4, help='cell_size, from 2 to 8, int')
    parser.add_argument('--model_name', type=str, default='YOUR_MODEL_NAME', required=True, help='160_net_G_A.pth should be in this directory.')
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output
    cell_size = args.cell_size
    use_cpu = args.cpu
    model_name = args.model_name

    if not os.path.exists("alias_net.pth"):
        print("missing models")

    # Load all required models
    print("Loading models...")
    binary_classifier = load_binary_classifier()
    pixel_model = Model(model_name, device="cpu" if use_cpu else "cuda")
    pixel_model.load()
    print("All models loaded successfully")

    pairs = []

    if os.path.isdir(in_path):
        in_images = glob.glob(in_path + "/*.png") + glob.glob(in_path + "/*.jpg")
        if not out_path:
            out_path = os.path.join(in_path, "outputs")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        elif os.path.isfile(out_path):
            print("output cant be a file if input is a directory")
            return
        for i in in_images:
            pairs += [(i, i.replace(in_path, out_path))]
    elif os.path.isfile(in_path):
        if not out_path:
            base, ext = os.path.splitext(in_path)
            out_path = base+"_pixelized"+ext
        else:
            if os.path.isdir(out_path):
                _, file = os.path.split(in_path)
                out_path = os.path.join(out_path, file)
        pairs = [(in_path, out_path)]

    for in_file, out_file in pairs:
        print(f"Processing {in_file}...")
        
        # Step 1: Check if image is real
        is_real = is_real_image(in_file, binary_classifier)
        print(f"Image is {'real' if is_real else 'unreal'}")
        
        # Step 2: For real images, apply anime generation first
        if is_real:
            print("Applying anime generation...")
            anime_file = pixel_model.animeize_image(in_file)
            pixelize_input = anime_file
        else:
            pixelize_input = in_file
        
        # Step 3: Apply pixelization
        print("Applying pixelization...")
        pixel_model.pixelize(pixelize_input, out_file, cell_size)
        
        # Clean up intermediate files
        if is_real and os.path.exists(anime_file):
            os.remove(anime_file)
        
        print(f"Completed processing {in_file} -> {out_file}")

if __name__ == "__main__":
    pixelize_cli()