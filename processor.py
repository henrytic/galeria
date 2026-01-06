import os
import shutil
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from PIL import Image

class FaceProcessor:
    def __init__(self, upload_folder='static/uploads', faces_folder='static/faces'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {self.device}")
        
        # Initialize MTCNN for face detection
        # Increased min_face_size to avoid very small background faces
        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device, keep_all=True
        )
        
        # Initialize Inception Resnet V1 for face recognition (embeddings)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.upload_folder = upload_folder
        self.faces_folder = faces_folder
        
        # In-memory storage for this demo
        self.data_records = [] 
        # Structure: {'image_path': str, 'face_crop_path': str, 'embedding': np.array, 'cluster_id': int}
        
        self.names = {} # Map cluster_id -> name

    def process_images(self):
        """
        Scans upload folder, detects faces, generates embeddings, clusters them.
        """
        # Clear previous run data (optional, for this demo logic)
        self.data_records = []
        if os.path.exists(self.faces_folder):
            shutil.rmtree(self.faces_folder)
        os.makedirs(self.faces_folder, exist_ok=True)

        image_files = [f for f in os.listdir(self.upload_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        all_embeddings = []
        temp_records = []

        print(f"Processing {len(image_files)} images...")

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(self.upload_folder, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Detect faces
                boxes, probs = self.mtcnn.detect(img)
                
                if boxes is not None:
                    # Get cropped faces for embedding generation
                    # mtcnn(img) returns a tensor of cropped faces if keep_all=True
                    faces_tensors = self.mtcnn(img) 
                    
                    if faces_tensors is not None:
                        # If only one face, it might not have the batch dim
                        if faces_tensors.ndim == 3:
                           faces_tensors = faces_tensors.unsqueeze(0)

                        # Generate embeddings
                        embeddings = self.resnet(faces_tensors.to(self.device)).detach().cpu().numpy()
                        
                        valid_face_count = 0
                        for i, box in enumerate(boxes):
                            # Filter by probability to remove blurry/uncertain faces
                            if probs[i] < 0.90:
                                continue
                                
                            # Save face crop for UI
                            face_crop_filename = f"face_{idx}_{i}.jpg"
                            face_crop_path = os.path.join(self.faces_folder, face_crop_filename)
                            
                            rotation = 0 # No rotation handling in basic MTCNN usage here
                            # Manual crop to ensure we have the image file
                            # box is [x1, y1, x2, y2]
                            crop_img = img.crop(box)
                            crop_img.save(face_crop_path)

                            emb = embeddings[i]
                            
                            temp_records.append({
                                'image_path': f'/static/uploads/{img_file}',
                                'face_crop_path': f'/static/faces/{face_crop_filename}',
                                'embedding': emb,
                                'probability': float(probs[i])
                            })
                            all_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        if not all_embeddings:
            print("No faces found.")
            return []

        # Clustering
        # Adjusted parameters: 
        # eps=0.85 (good for grouping variations)
        # min_samples=3 (filters out faces appearing less than 3 times, effectively removing noise/one-offs)
        clustering = DBSCAN(eps=0.85, min_samples=2, metric='euclidean').fit(all_embeddings)
        labels = clustering.labels_

        # Assign labels back to records
        for i, record in enumerate(temp_records):
            record['cluster_id'] = int(labels[i])
            self.data_records.append(record)
            
        return self.get_people_summary()

    def get_people_summary(self):
        """
        Returns a dictionary or list used by the API.
        """
        people = {}
        for record in self.data_records:
            cid = record['cluster_id']
            if cid == -1:
                continue # Noise, ignore or put in 'unknown'
            
            name = self.names.get(cid, f"Persona {cid}")
            
            if cid not in people:
                people[cid] = {
                    'id': cid,
                    'name': name,
                    'face_url': record['face_crop_path'], # Use first face as thumbnail
                    'images': set()
                }
            people[cid]['images'].add(record['image_path'])
        
        # Convert sets to lists
        result = []
        for cid, data in people.items():
            data['images'] = list(data['images'])
            result.append(data)
        
        return result

    def get_gallery_images(self):
        # Just return all uploaded images
        files = [f for f in os.listdir(self.upload_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return [f'/static/uploads/{f}' for f in files]

    def get_person_images(self, person_id):
        images = set()
        for record in self.data_records:
            if record['cluster_id'] == int(person_id):
                images.add(record['image_path'])
        return list(images)

    def rename_person(self, person_id, name):
        self.names[int(person_id)] = name
        return True

    def get_metrics(self):
        """
        Calculates and returns internal model metrics.
        """
        if not self.data_records:
            return {
                'total_faces': 0,
                'total_people': 0,
                'noise_faces': 0,
                'avg_confidence': 0,
                'silhouette_score': 0
            }
            
        embeddings = [r['embedding'] for r in self.data_records]
        labels = [r['cluster_id'] for r in self.data_records]
        probs = [r['probability'] for r in self.data_records]
        
        # Count unique labels (excluding -1 for noise)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        total_people = len(unique_labels)
        noise_faces = labels.count(-1)
        total_faces = len(labels)
        avg_confidence = np.mean(probs) if probs else 0
        
        # Silhouette Score (requires at least 2 clusters or 1 cluster and noise)
        # Note: Scikit-learn says: "The Silhouette Coefficient is defined for 2 <= n_labels <= n_samples - 1."
        try:
            if len(set(labels)) > 1:
                sil_score = silhouette_score(embeddings, labels, metric='euclidean')
                db_score = davies_bouldin_score(embeddings, labels)
                ch_score = calinski_harabasz_score(embeddings, labels)
            else:
                sil_score = 0.0
                db_score = 0.0
                ch_score = 0.0
        except:
            sil_score = 0.0
            db_score = 0.0
            ch_score = 0.0

        return {
            'total_faces': total_faces,
            'total_people': total_people,
            'noise_faces': noise_faces,
            'avg_confidence': round(avg_confidence * 100, 2),
            'silhouette_score': round(sil_score, 4),
            'davies_bouldin': round(db_score, 4),
            'calinski_harabasz': round(ch_score, 4)
        }

    def get_scatter_data(self):
        """
        Reduces embeddings to 2D for visualization.
        """
        if not self.data_records or len(self.data_records) < 2:
            return []

        embeddings = [r['embedding'] for r in self.data_records]
        labels = [r['cluster_id'] for r in self.data_records]
        paths = [r['face_crop_path'] for r in self.data_records]
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        data = []
        for i, (x, y) in enumerate(coords):
            cid = int(labels[i])
            name = "Ruido" if cid == -1 else self.names.get(cid, f"Persona {cid}")
            
            data.append({
                'x': float(x),
                'y': float(y),
                'cluster': cid,
                'label': name,
                'face': paths[i]
            })
        return data
