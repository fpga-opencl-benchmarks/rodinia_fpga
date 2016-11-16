/* ============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct{
	int starting;
	int no_of_edges;
} Node;

//--7 parameters
__kernel void BFS_1(__global const Node* restrict g_graph_nodes,
		    __global const int*  restrict g_graph_edges, 
		    __global char*       restrict g_graph_mask, 
		    __global char*       restrict g_updating_graph_mask, 
		    __global char*       restrict g_graph_visited, 
		    __global int*        restrict g_cost, 
		             const int   no_of_nodes)
{
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		int cost = g_cost[tid]+1;
		Node graph_node = g_graph_nodes[tid];
//		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
#pragma ivdep
#pragma unroll 2
		for(int i=graph_node.starting; i<(graph_node.no_of_edges + graph_node.starting); i++)
		{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
			{
				g_cost[id]=cost;
				g_updating_graph_mask[id]=true;
			}
		}
	}
}

//--5 parameters
__attribute__((num_simd_work_items(16)))
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void BFS_2(__global char*     restrict g_graph_mask, 
		    __global char*     restrict g_updating_graph_mask, 
		    __global char*     restrict g_graph_visited, 
		    __global char*     restrict g_over,
		             const int no_of_nodes)
{
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_updating_graph_mask[tid])
	{
		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}


