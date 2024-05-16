# make cosivility graph
from co_visibility_graph_vec import *

# 사용 예시

def main_vec():
    a = torch.FloatTensor([1,2,3,3,2,4]) 
    b = torch.FloatTensor([1,2,3,2,3,2])  
    c = torch.FloatTensor([2,1,3,3,2,3])
    d = torch.FloatTensor([2,1,4,6,2,2])

    co_visibility_graph = build_co_visibility_graph([1,2,3,4], [a,b,c,d])
    mst = kruskal_mst(co_visibility_graph)
    print("existing edges" ,mst.get_edges())

    e = torch.FloatTensor([3,4,5,6,7,8])
    add_image_to_mst(mst, 5, e)
    print("new edges" , mst.get_edges())


main_vec()
