from numpy import ndarray

def clean_string(string: str, substrings: list):
    for substring in substrings:
        string = string.replace(substring, "")
    return string

def remove_newline(string: str):
    return string.replace("\n", "")

def tabular_data(headers: list, 
                 items: list, 
                 data: ndarray, 
                 width: int = None) -> str:
    """Returns a properly formatted string that contains
    ------------------------------------------------------------------
    |            | headers[0] | headers[1] |     ...    | headers[m] |
    ------------------------------------------------------------------
    |  items[0]  |  data[0,0] |  data[0,1] |     ...    |  data[0,m] |
    ------------------------------------------------------------------
    |  items[1]  |  data[1,0] |  data[1,1] |     ...    |  data[1,m] |
    ------------------------------------------------------------------
    |     ...    |     ...    |     ...    |     ...    |     ...    |
    ------------------------------------------------------------------
    |  items[n]  |  data[n,0] |  data[n,1] |     ...    |  data[n,m] |
    ------------------------------------------------------------------

    Parameters
    ----------
    headers list[str]  : List with the names of the columns of the table
    items   list[str]  : List with the names of the rows    of the table
    data    np.ndarray : Array of shape (len(items), len(headers))
    width   int        : Desired width of the cell of the table. If the 
                         width specified is smaller than the minimum cell
                         width necessary for correct display, this value
                         is overwritten.
    """
    
    # Check if the table can be printed in the first place
    rows = len(items)
    cols = len(headers)
    if data.shape != (rows, cols):
        raise(ValueError, f'data shape {data.shape} incompatible with headers {cols} x items {rows}')
    
    # Find the minimum cell width for correct display
    min_width = max(max([len(header) for header in headers]), 
                    max([len(item) for item in items]))
    
    if width is None or width < min_width:
        width = min_width
    
    # The width of a column is width + 2 ' ' + 1 '|' character. 
    # The total number of columns is given by the headers + 1
    # to display the items
    # At the end add 1 character for the final '|'.
    horizontal_line = "-"*((width + 3)*(cols + 1) + 1) + "\n"
    
    # Format header row with the titles of the columns
    header_string   = f"| {'':^{width}} "
    for header in headers:
        header_string += f"| {header:^{width}} "
    header_string += "|\n"
    
    # Start building the entire string
    to_print = horizontal_line + header_string + horizontal_line 
    
    for i, row in enumerate(items):
        row_string = f"| {row:^{width}} "
        
        for j in range(cols):
            row_string += f"| {data[i,j]:{width}.2f} "
            
        row_string += "|\n"
        to_print += row_string + horizontal_line
    
    return to_print