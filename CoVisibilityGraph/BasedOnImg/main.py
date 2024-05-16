# make cosivility graph
from co_visibility_graph import *

# 사용 예시
import os
import sys

# 초기 이미지 파일 불러오기
image_paths = list(map(lambda x : "./imgs/" + x, os.listdir("./imgs")))
existing_images = image_paths[-4:] 
print(len(existing_images)) # 4개

# co_visibility_graph build
co_visibility_graph = build_co_visibility_graph(existing_images) # 여기서 오래 걸림.
mst = kruskal_mst(co_visibility_graph)
print("existing edges" ,mst.get_edges())

# 새로운 이미지 추가
new_image_path = image_paths[0] # 1개
add_image_to_mst(mst, new_image_path)
print("new edges" ,mst.get_edges())

# 특정 이미지 쌍의 공통 특징점 수 확인
common_points = mst.get_common_points(existing_images[2], existing_images[-1])
print(f"Image1 and Image4 have {common_points} common points in the MST.")


