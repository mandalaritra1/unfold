
import numpy as np



class bin_edges:
    def __init__(self, groomed):
        self.groomed = groomed
        self.pt_edges = [200, 290, 400, 13000]

        if not self.groomed:
            self.edges = [
                20, 30, 40,
                50, 60, 70, 80, 90, 100,
                125, 150, 175, 200, 500, 13000
            ]
            
            self.edges_gen = [
                20, 40, 60, 80, 100,
                150, 200, 13000
            ]
            self.mass_edges_reco = [20,30,40,50,60,70,80,90,100,125,150,175,200,500,13000]
            self.mass_edges_gen  = [20,40,60,80,100,150,200,13000]
            
            
            self.reco_mass_edges_by_pt = [[20,30,40,50,60,70,80,100,13000], #pt bin 1
                                    [20,30,40,50,60,70,80,90,100,150,13000], #pt bin 2
                                    [20,30,40,50,60,70,80,90,100,125,150,175,200,500,13000] #pt bin 3
                                    ]

            self.gen_mass_edges_by_pt = [[20,40,60,80,13000], #pt bin 1
                                    [20,40,60,80,100,13000], #pt bin 2
                                    [20,40,60,80,100,150,200,13000] #pt bin 3
                                ]


        
        if self.groomed:
            self.edges = [
                0, 10,  20, 30, 40,
                50, 60, 70, 80, 90, 100,
                125, 150, 175, 200, 500, 13000
            ]
            
            self.edges_gen = [
                0, 10, 20, 40, 60, 80, 100,
                150, 200, 13000
            ]

            self.mass_edges_reco = [0,10,20,30,40,50,60,70,80,90,100,125,150,175,200,500,13000]
            self.mass_edges_gen  = [0,10,20,40,60,80,100,150,200,13000]
            
            
        
            self.reco_mass_edges_by_pt = [[0,10,20,30,40,50,60,70,80,90,100,150,13000], #pt bin 1
                                    #[0,10,20,30,40,50,60,70,80,90,100,150, 200, 500, 13000], #pt bin 2
                                    [0,10,20,30,40,50,60,80,100,150, 200, 500, 13000], #pt bin 2
                                    [0,10,20,30,40,50,60,80,100,125,150,175,200,500,13000] #pt bin 3
                                    ]

            self.gen_mass_edges_by_pt = [[0,10,20,40,60,80,100,13000], #pt bin 1
                                    #[0,10,20,40,60,80,100, 150, 13000], #pt bin 2
                                    [0,10,20,40,60,100, 150, 13000], #pt bin 2
                                    [0,10,20,40,60,100,150,200,13000] #pt bin 3
                                ]
    

        

def compress_open_ended_last_bin(edges, *, match_to_second_last=True):
    """
    Return a copy of `edges` where the very last (open-ended) upper edge is replaced
    by a finite value so that the last bin's *width* is similar to the second-last bin.

    Example:
        [20, 30, 40, ..., 200, 500, 13000]  -->  [20, 30, 40, ..., 200, 500, 800]
        because second-last width = 500-200 = 300, so last bin becomes 500+300.

    Notes:
    - We do NOT throw away overflow content. This is just for *display/geometry*.
      In plots/hist stairs we'll use these compressed edges while the content stays
      the same (overflow included in the last bin).

    Parameters
    ----------
    edges : sequence of float
    match_to_second_last : bool
        If True, make last width == second-last width.
        If False, make it minimum of the previous two widths (very similar).

    Returns
    -------
    new_edges : np.ndarray
    """
    e = np.asarray(edges, dtype=float).copy()
    if len(e) < 3:
        return e

    w1 = e[-1] - e[-2]  # typically "too large" (e.g. 13000-500)
    w2 = e[-2] - e[-3]  # the second-last width

    if match_to_second_last or w2 <= 0:
        new_last = e[-2] + abs(w2)
    else:
        # make it close to previous width but not larger
        new_last = e[-2] + max(1.0, min(abs(w1), abs(w2)))

    e[-1] = new_last
    return e

def compress_edges_by_pt(edge_lists):
    """Apply compress_open_ended_last_bin to each edge list in a list-of-lists."""
    return [compress_open_ended_last_bin(edges) for edges in edge_lists]
