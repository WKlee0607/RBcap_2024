import cv2
import numpy as np


class CoVisibilityGraph:
    def __init__(self):
        self.graph = {}  # 노드를 키로, 엣지를 값으로 갖는 딕셔너리
    
    def add_image(self, image_id):
        if image_id not in self.graph:
            self.graph[image_id] = {}

    def add_edge(self, image_id1, image_id2, common_points):
        if image_id1 not in self.graph:
            self.add_image(image_id1)
        if image_id2 not in self.graph:
            self.add_image(image_id2)
        
        self.graph[image_id1][image_id2] = common_points
        self.graph[image_id2][image_id1] = common_points

    def get_common_points(self, image_id1, image_id2):
         return self.graph[image_id1].get(image_id2, None)
    
    def get_edges(self): 
        edges = {} # {(img1, img2): common_points, ...}
        for image_id1 in self.graph:
            for image_id2 in self.graph[image_id1]:
                if (image_id2, image_id1) not in edges.keys(): # 반대되는 edge가 있으면 컷
                    common_points = self.graph[image_id1][image_id2]
                    edges[(image_id1, image_id2)] = common_points
        return edges
    
    def get_nodes(self):
        return list(self.graph.keys())
    

def extract_features(image_path):
    orb = cv2.ORB_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def build_co_visibility_graph(image_paths:list):
    co_visibility_graph = CoVisibilityGraph()
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            image_id1 = image_paths[i]
            image_id2 = image_paths[j]
            kp1, descriptors1 = extract_features(image_id1)
            kp2, descriptors2 = extract_features(image_id2)
            matches = match_features(descriptors1, descriptors2)
            common_points = len(matches)
            if common_points > 0:
                co_visibility_graph.add_edge(image_id1, image_id2, common_points)
    return co_visibility_graph

def kruskal_mst(co_visibility_graph):
    parent = {}
    rank = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    
    def union(node1, node2):
        root1 = find(node1) # node 반환
        root2 = find(node2) # node 반환
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1
    
    edges = co_visibility_graph.get_edges() # ok
    sorted_edges = sorted(edges.items(), key = lambda item: item[1], reverse=True) # [(key, value), (key, value)] <= 내림차순    

    for node in co_visibility_graph.graph: # 초기화
        parent[node] = node # node 개수
        rank[node] = 0

    mst = CoVisibilityGraph()


    for edge in sorted_edges:
        print("edge", edge)
        (image_id1, image_id2), common_points = edge
        if find(image_id1) != find(image_id2):
            union(image_id1, image_id2)
            mst.add_edge(image_id1, image_id2, common_points)

    return mst

def add_image_to_mst(mst, new_image_path):
    keypoints_new, descriptors_new = extract_features(new_image_path)
    new_image_id = new_image_path

    best_edges = {}
    for existing_image_id in mst.get_nodes():
        keypoints_existing, descriptors_existing = extract_features(existing_image_id)
        matches = match_features(descriptors_new, descriptors_existing)
        common_points = len(matches)
        if common_points > 0:
            best_edges[(new_image_id, existing_image_id)] = common_points
    
    sorted_best_edges = sorted(best_edges.items(), key = lambda item: item[1], reverse=True)

    parent = {}
    rank = {}
    for node in mst.get_nodes() + [new_image_id]:
        parent[node] = node
        rank[node] = 0

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    for edge in sorted_best_edges:
        (image_id1, image_id2), common_points = edge
        if find(image_id1) != find(image_id2):
            union(image_id1, image_id2)
            mst.add_edge(image_id1, image_id2, common_points)
            break
