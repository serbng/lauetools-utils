def grid_chunks(num_rows, num_cols, chunk_size):
    """
    Suddivide una griglia di dimensioni num_rows x num_cols
    in sotto-griglie (chunk) di dimensione chunk_size x chunk_size (circa).
    Restituisce una lista di "chunk", dove ciascun chunk è 
    una lista di indici lineari (r * num_cols + c) 
    corrispondenti alle celle (r, c) del sotto-blocco.
    
    Esempio: chunk_size=10 -> blocchi 10x10,
    l'ultimo blocco può essere più piccolo se 81 non è un multiplo di 10.
    """
    chunks = []
    for r_start in range(0, num_rows, chunk_size):
        r_end = min(r_start + chunk_size, num_rows)
        for c_start in range(0, num_cols, chunk_size):
            c_end = min(c_start + chunk_size, num_cols)
            
            # Costruiamo la lista di indici "lineari" di questo sotto-blocco
            subgrid_indices = []
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    idx = r * num_cols + c
                    subgrid_indices.append(idx)
            
            chunks.append(subgrid_indices)
    
    return chunks

def linear_chunks(total, chunk_size):
    indices = list(range(total))
    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = indices[start:end]
        chunks.append(chunk)
    return chunks