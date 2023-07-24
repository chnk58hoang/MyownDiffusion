from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import pandas as pd
import glob


class CustomCifar10(Dataset):
    def __init__(self,data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.all_data_folders = os.listdir(self.data_dir)
        dataframe = pd.DataFrame()
        classes = []
        img_paths = []
        for i,folder_name in enumerate(self.all_data_folders):
            img_path = glob.glob(os.path.join(self.data_dir,folder_name,'*.*'))
            img_paths += img_path
            classes += [i]*len(img_path)
        
        dataframe['classes'] = classes
        dataframe['img_paths'] = img_paths

        self.dataframe = dataframe

        self.transform = transforms.Compose([
                                    transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
                                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])


    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        image_path = self.dataframe['img_paths'].iloc[index]
        cls = self.dataframe['classes'].iloc[index]   

        image = Image.open(image_path)
        image = self.transform(image)

        return image,cls






