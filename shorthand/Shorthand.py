import pandas as pd
import shorthand as shnd
from pathlib import Path


def _create_id_map(domain, drop_na=True, **kwargs):
    '''
    Maps distinct values in a domain to a range of integers.  Additional
    keyword arguments are passed to the pandas.Series constructor when
    the map series is created.

    Parameters
    ----------
    domain : list-like (coercible to pandas.Series)
        Arbitrary set of values to map. May contain duplicates.

    drop_na : bool, default True
        Ignore null values and map only non-null values to integers.

    Returns
    -------
    pandas.Series
        Series whose length is the number of distinct values in the
        input domain.

    Examples
    --------
    >>> import pandas as pd
    >>> dom = ['a', 'a', 'b', pd.NA, 'f', 'b']
    >>> _create_id_map(dom, dtype=pd.UInt32Dtype())

    a    0
    b    1
    f    2
    dtype: UInt32

    >>> _create_id_map(dom, drop_na=False, dtype=pd.UInt32Dtype())

    a       0
    b       1
    <NA>    2
    f       3
    dtype: UInt32
    '''
    # check if domain object has a str attribute like a pandas.Series
    # and convert if not
    try:
        assert domain.str
        # make a copy so we can mutate one (potentially large) object
        # instead of creating additional references
        domain = domain.copy()
    except AttributeError:
        domain = pd.Series(domain)

    if drop_na:
        domain = domain.loc[~domain.isna()]

    distinct_values = domain.unique()

    id_map = pd.Series(
        range(len(distinct_values)),
        index=distinct_values,
        **kwargs
    )

    return id_map


def _strip_csv_comments(column, pattern):

    column = column.str.split(pat=pattern, expand=True)
    return column[0]


def _expand_items(
    group,
    entry_syntax,
    entry_prefix_id_map,
    item_label_id_map
):
    '''
    THIS FUNCTION MUTATES ITS FIRST ARGUMENT

    Takes parsed shorthand entries grouped by entry prefix and item
    label, checks if the entry syntax has a list delimiter for this
    item, and splits the string on the delimiter if it exists.
    '''

    # The group name is a tuple (entry_prefix_id, item_label_id)
    if pd.isna(group.name[1]):
        # If the item label ID is NA then this group is the entry
        # strings rather than items parsed out of the entries
        return group

    else:
        # Get the item label ID
        item_label_id = group.name[1]

    # Locate this item label ID in item_label_id_map and get the string
    # value for the item label out of the map index
    item_label = item_label_id_map.loc[
        item_label_id_map == item_label_id
    ]
    item_label = item_label.index.drop_duplicates()
    if len(item_label) > 1:
        raise ValueError('item label ID not unique')
    else:
        item_label = item_label[0]

    if pd.isna(item_label):
        # If nulls were accounted for in the label ID map and the label
        # we just recovered is null, then this group is the entry
        # strings rather than items parsed out of the entries
        return group

    # Locate this entry prefix ID in entry_prefix_id and get the string
    # value for the entry prefix out of the map index
    entry_prefix_id = group.name[0]
    entry_prefix = entry_prefix_id_map.loc[
        entry_prefix_id_map == entry_prefix_id
    ]
    entry_prefix = entry_prefix.index.drop_duplicates()
    if len(entry_prefix) > 1:
        raise ValueError('entry prefix ID not unique')
    else:
        entry_prefix = entry_prefix[0]

    # Locate the row for this item in the entry syntax and get the
    # list delimiter
    item_syntax = entry_syntax.query('entry_prefix == @entry_prefix') \
                              .query('item_label == @item_label') \
                              .squeeze()
    delimiter = item_syntax['list_delimiter']

    if pd.notna(delimiter):
        # If there is a delimiter, split this group's strings in place
        group.loc[:, 'string'] = group['string'].str.split(delimiter)
        return group

    else:
        # If there is no delimiter, return the group
        return group


def _normalize_shorthand(shnd_input, comment_char, fill_cols, drop_na):
    '''
    Fill or drop missing values in shorthand input and parse comments.
    Comments have to be parsed in this function rather than using
    pd.read_csv(comment=comment_char, escapechar='\') because there may
    be escaped characters in entries that should be parsed separately.
    Using read_csv with an escape character removes the escape character
    anywhere in the file, so the non-comment character escapes would be
    lost.

    The input file must have a header and the first four labels of the
    header must be (in any order and any character case):
        [left_entry, right_entry, link_tags_or_override, reference]

    Parameters
    ----------
    shnd_input : pandas.DataFrame
        Unparsed shorthand data

    skiprows : int
        Number of lines to skip at the beginning of the input file.

    comment_char : str
        Character indicating the rest of a line should be skipped. Must
        be a single character.

    fill_cols : scalar or non-string iterable, default 'left_entry'
        Label(s) of columns that will be forward filled.

    drop_na : scalar or non-string iterable, default 'right_entry'
        Column labels. Rows will be dropped if they have null values in
        columns these columns.

    Returns
    -------
    pandas.DataFrame
        Normalized shorthand input data
    '''

    if len(shnd_input.columns) > 256:
        raise ValueError(
            'Shorthand input csv files cannot have more than 256 '
            'columns'
        )

    required_cols = ['left_entry',
                     'right_entry',
                     'link_tags_or_override',
                     'reference']

    valid_columns = [
        (c in map(str.casefold, shnd_input.columns)) for c in required_cols
    ]

    if not all(valid_columns):
        raise ValueError(
            'shorthand csv files must have a header whose first four '
            'column labels must be (ignoring case and list order):\n'
            '>>> ["left_entry", "right_entry", '
            '"link_tags_or_override", "reference"]'
        )

    # If the comment character is a regex metacharacter, escape it
    comment_char = shnd.util.escape_regex_metachars(comment_char)
    # Regular expressions to match bare and escaped comment characters
    unescaped_comment_regex = r"(?<!\\)[{}]".format(comment_char)
    escaped_comment_regex = fr"(\\{comment_char})"

    # Find cells where comments start
    has_comment = shnd_input.apply(
        lambda x: x.str.contains(unescaped_comment_regex)
    )
    has_comment = has_comment.fillna(False)

    # Set cells to the right of cells with comments to False because
    # they were created by commas in comments
    commented_out = has_comment.where(has_comment).ffill(axis=1).fillna(False)
    commented_out = commented_out ^ has_comment

    # shnd_input = _set_StringDtype(shnd_input.mask(commented_out)))
    # can't use pd.StringDtype() throughout because it currently doesn't
    # allow construction with null types other than pd.NA. This will
    # likely change soon
    # https://github.com/pandas-dev/pandas/pull/41412

    # Mask off cells to the right of cells with comments in them
    shnd_input = shnd_input.mask(commented_out)

    # Split cells where comments start and take the uncommented part
    has_comment = has_comment.any(axis=1)
    shnd_input.loc[has_comment, :] = shnd_input.loc[has_comment].apply(
        _strip_csv_comments,
        args=(comment_char,)
    )

    # Drop rows that began with comments
    shnd_input = shnd_input.mask(shnd_input == '')
    shnd_input = shnd_input.dropna(how='all')

    # Replace escaped comment characters with bare comment characters
    shnd_input = shnd_input.apply(
        shnd.util.replace_escaped_comment_chars,
        args=(comment_char, escaped_comment_regex)
    )

    # Optionally forward fill missing values
    for column in fill_cols:
        shnd_input.loc[:, column] = shnd_input.loc[:, column].ffill()

    # Optionally drop lines missing values
    shnd_input = shnd_input.dropna(subset=drop_na)

    return shnd_input


def _get_item_link_source_IDs(group):
    '''
    Take a group of items that all have the same csv row and column
    index, meaning they were exploded out of a single entry, and return
    the string ID for the entry they were exploded from.

    If the entry syntax indicated that there are no links between a
    group of item strings and the entry string that contained them,
    return null values
    '''
    link_type_is_na = group['link_type_id'].isna()

    # If all the link types are NA then there are no links between items
    # inside this entry and the entry string itself
    if link_type_is_na.all():
        return pd.Series([pd.NA]*len(group))

    # If any link types are not null then return the string_id of the
    # entry the items were exploded from
    else:
        entry_string_id = group.loc[link_type_is_na, 'string_id'].squeeze()
        return pd.Series([entry_string_id]*len(group))


def _get_entry_prefix_ids(
    side,
    shnd_data,
    dplct_entries,
    csv_column_id_map
):
    '''
    Reconstruct entry prefixes for one side of a link between
    entries, including duplicates not present in data
    '''

    # "right" and "left" entries are defined in the link syntax
    # file, but they don't constrain the location of those
    # columns in the input csv file, so we need to get the index
    # of the csv column for this side of the link.
    csv_col = csv_column_id_map[side + '_entry']
    label = side + '_entry_prefix_id'
    entry_prefixes = shnd_data.loc[
        :,
        ['csv_row', 'csv_col', 'entry_prefix_id', 'item_label_id']
    ]

    # Locate entry strings by finding null item labels
    entry_prefixes = entry_prefixes.query('item_label_id.isna()') \
                                   .query('csv_col == @csv_col')
    # The item label column is now null values, so drop it
    entry_prefixes = entry_prefixes.drop('item_label_id', axis='columns')

    # "string" csv indexes locating distinct strings are
    # different from "entry" csv indexes, which locate entries
    # with potentially duplicate string values. Copy the string
    # csv indexes to fill in missing data after merger with
    # duplicates.
    entry_prefixes = entry_prefixes.rename(
        columns={
            'entry_prefix_id': label,
            'csv_row': 'string_csv_row',
            'csv_col': 'string_csv_col'
        }
    )

    # Get csv indexes for duplicate entries on this side of the
    # link and merge the map of duplicates with entry_prefixes
    # to match the csv indexes of distinct strings with csv
    # indexes of duplicate strings
    on_side_dplcts = dplct_entries.query('entry_csv_col == @csv_col')
    on_side_dplcts = entry_prefixes.merge(on_side_dplcts, how='right')

    # Collect all the prefixes together
    entry_prefixes = pd.concat([entry_prefixes, on_side_dplcts])

    # The "entry" csv indexes for the distinct strings are now
    # null, so copy the "string" csv indexes over to the "entry"
    # columns
    missing_entry = entry_prefixes['entry_csv_row'].isna()

    string_csv_idx = entry_prefixes.loc[
        missing_entry,
        ['string_csv_row', 'string_csv_col']
    ]
    string_csv_idx = string_csv_idx.to_numpy()

    entry_csv_cols = ['entry_csv_row', 'entry_csv_col']
    entry_prefixes.loc[missing_entry, entry_csv_cols] = string_csv_idx

    return entry_prefixes


def _get_link_component_string_ids(
        prefix_pairs,
        shnd_data,
        link_syntax,
        component_prefix,
        subset,
        columns=None,
        prefix_string_id=True
):
    '''
    Links have three string components: source, target, and
    reference. This function locates the data for one
    component in the link syntax and merges the position
    code for that component with a dataframe of entry prefix
    pairs to get all data required to locate the string ID
    for each link component.

    Parameters
    ----------

    link_syntax: pandas.DataFrame
        data defining links between pairs of shorthand entries

    component_prefix: str
        One of ['src_', 'tgt_', 'ref_']

    subset : pandas.Series
        A row indexer to subset the link syntax.

    include_link_id : bool, default False
        If True, select link type IDs from the mutable data

    columns : list-like or None, default None
        Columns to include in the returned dataframe. If None,
        return all columns.

    prefix_string_id : bool, default True
        Optionally add the component prefix to the string ID
        column label before returning.
    '''

    link_syntax_selection = link_syntax.loc[subset]

    columns_for_this_component = [
        'left_entry_prefix_id',
        'right_entry_prefix_id',
        component_prefix + 'csv_col',
        component_prefix + 'item_label',
        'link_type_id'
    ]

    # The reference position code could be null if the reference
    # string should be the input file, so drop any rows in the
    # link syntax that have no value for the current position
    # code
    component_csv_col = component_prefix + 'csv_col'
    link_syntax_selection = link_syntax_selection.dropna(
        subset=component_csv_col
    )

    # Merge the prefix pairs with the link syntax to get csv
    # rows and item labels for every link between entries
    link_component = prefix_pairs.merge(
        link_syntax_selection[columns_for_this_component],
        on=['left_entry_prefix_id', 'right_entry_prefix_id']
    )

    # Locate rows for which the link syntax says this component
    # (source, target, or reference) is in the left csv column
    is_L = (link_component[component_prefix + 'csv_col'] == 'l')
    # Extract the csv rows and columns that locate the string
    # value in the parsed data
    left_indexes = link_component[['L_str_csv_row', 'L_str_csv_col']]
    left_indexes = left_indexes.loc[is_L]
    left_indexes.columns = ['csv_row', 'csv_col']

    # Locate rows for which the link syntax says this component
    # (source, target, or reference) is in the right csv column
    is_R = (link_component[component_prefix + 'csv_col'] == 'r')
    # Extract the csv rows and columns that locate the string
    # value in the parsed data
    right_indexes = link_component[['R_str_csv_row', 'R_str_csv_col']]
    right_indexes = right_indexes.loc[is_R]
    right_indexes.columns = ['csv_row', 'csv_col']

    # Combine the csv indexes above into a single dataframe and
    # overwrite the csv rows and columns defined by the link
    # syntax with indexes that locate string values in the
    # parsed data
    link_component[['csv_row', 'csv_col']] = pd.concat(
        [left_indexes, right_indexes]
    ).sort_index()

    # Drop all the information required to sort out which column
    # the string values came from
    link_component = link_component.drop(
        ['left_entry_prefix_id',
         'right_entry_prefix_id',
         'L_str_csv_row',
         'L_str_csv_col',
         'R_str_csv_row',
         'R_str_csv_col',
         component_csv_col],
        axis='columns'
    )

    # Rename the item label column so we can merge the link
    # component with the parsed data and recover the string IDs
    link_component = link_component.rename(
        columns={component_prefix + 'item_label': 'item_label_id'}
    )

    # The link component dataframe now contains the three pieces
    # of information we need to locate string IDs in the mutated
    # input data: csv row, csv column, and item label. We select
    # those columns from the mutable data along with the item
    # list positions so links can be tagged with positions in
    # ordered lists.
    data_columns = [
        'csv_row',
        'csv_col',
        'item_label_id',
        'item_list_position',
        'string_id'
    ]

    link_component = link_component.merge(
        shnd_data[data_columns],
        on=['csv_row', 'csv_col', 'item_label_id'],
        how='left'
    )

    # Optionally clean up the output

    if columns is not None:
        link_component = link_component[columns]

    if prefix_string_id and ('string_id' in columns):
        if isinstance(link_component, pd.DataFrame):
            link_component = link_component.rename(
                columns={'string_id': component_prefix + 'string_id'}
            )
        else:
            link_component = link_component.rename(
                component_prefix + 'string_id'
            )

    return link_component


def _copy_cross_duplicates(L_prefixes, R_prefixes):
    '''
    THIS FUNCTION MUTATES BOTH ARGUMENTS

    Get prefixes for duplicate entries in one column whose
    corresponding distinct string values are present in the
    other column
    '''

    L_missing_pfix = L_prefixes['left_entry_prefix_id'].isna()
    L_subset = [
        'string_csv_col', 'string_csv_row', 'left_entry_prefix_id'
    ]

    R_missing_pfix = R_prefixes['right_entry_prefix_id'].isna()
    R_subset = [
        'string_csv_col', 'string_csv_row', 'right_entry_prefix_id'
    ]

    # copy values from left to right
    to_copy = L_prefixes[L_subset].drop_duplicates()
    to_fill = R_prefixes.loc[R_missing_pfix, R_subset]

    cross_dplcts = to_copy.merge(
        to_fill,
        on=['string_csv_row', 'string_csv_col'],
        how='right'
    )
    cross_dplcts = cross_dplcts['left_entry_prefix_id'].array

    R_prefixes.loc[
        R_missing_pfix,
        'right_entry_prefix_id'
    ] = cross_dplcts

    # copy values from right to left
    to_copy = R_prefixes[R_subset].drop_duplicates()
    to_fill = L_prefixes.loc[L_missing_pfix, L_subset]

    cross_dplcts = to_copy.merge(
        to_fill,
        on=['string_csv_row', 'string_csv_col'],
        how='right'
    )
    cross_dplcts = cross_dplcts['right_entry_prefix_id'].array

    L_prefixes.loc[
        L_missing_pfix,
        'left_entry_prefix_id'
    ] = cross_dplcts


class Shorthand:
    '''
    A Shorthand has syntax definitions and provides methods that parse
    text according to the syntax.
    '''

    def __init__(
        self,
        entry_syntax,
        link_syntax=None,
        item_separator=None,
        default_entry_prefix=None,
        space_char=None,
        na_string_values=['!'],
        na_node_type='missing',
        syntax_case_sensitive=True
    ):

        try:
            # try and read it like a file stream
            self.entry_syntax = entry_syntax.read()

        except AttributeError:
            # otherwise assume it's a path
            with open(entry_syntax, 'r') as f:
                self.entry_syntax = f.read()

        self.syntax_case_sensitive = syntax_case_sensitive

        # Validate the entry syntax
        entry_syntax = shnd.syntax_parsing.validate_entry_syntax(
            self.entry_syntax,
            case_sensitive=syntax_case_sensitive
        )

        msg = (
            'Column "{}" not found in entry syntax. Must use the "{}" '
            'keyword when calling Shorthand().'
        )

        if item_separator is None:
            if 'item_separator' not in entry_syntax.columns:
                raise ValueError(msg.format('item_separator'))

            item_separator = shnd.util.get_single_value(
                entry_syntax,
                'item_separator'
            )

        self.item_separator = item_separator

        if default_entry_prefix is None:
            if 'default_entry_prefix' not in entry_syntax.columns:
                raise ValueError(msg.format('default_entry_prefix'))

            default_entry_prefix = shnd.util.get_single_value(
                entry_syntax,
                'default_entry_prefix'
            )

        self.default_entry_prefix = default_entry_prefix

        if link_syntax is not None:

            try:
                # try and read it like a file stream
                self.link_syntax = link_syntax.read()

            except AttributeError:
                # otherwise assume it's a path
                with open(link_syntax, 'r') as f:
                    self.link_syntax = f.read()

            # Validate the link syntax to raise any errors now without
            # storing the validated data
            shnd.syntax_parsing._validate_link_syntax(
                self.link_syntax,
                entry_syntax,
                case_sensitive=self.syntax_case_sensitive
            )

        if space_char is not None:
            space_char = str(space_char)

            if len(space_char) > 1:
                raise ValueError('space_char must be a single character')

        self.space_char = space_char

        if shnd.util.iterable_not_string(na_string_values):
            self.na_string_values = na_string_values
        else:
            self.na_string_values = [na_string_values]

        self.na_node_type = na_node_type

    def _apply_syntax(
        self,
        filepath_or_buffer,
        skiprows,
        comment_char,
        fill_cols,
        drop_na,
        big_id_dtype,
        small_id_dtype,
        list_position_base
    ):
        '''
        Takes a file-like object and parses it according to the
        definitions in Shorthand.entry_syntax and Shorthand.link_syntax.


        Parameters
        ----------
        filepath_or_buffer : str, path object, or file-like object
            A path or file-like object that returns csv-formatted text.
            "Path" is broadly defined and includes URLs. See
            pandas.read_csv documentation for details.

        skiprows : list-like, int or callable
            If int, number of lines to skip at the beginning of the
            input file. If list-like, 0-indexed set of line numbers to
            skip. See pandas.read_csv documentation for details.

        comment_char : str
            Indicates the remainder of a line should not be parsed.

        fill_cols : scalar or non-string iterable
            Label(s) of columns in the input file to forward fill.

        drop_na : scalar or non-string iterable
            Column labels. Rows will be dropped if they have null values
            in these columns.

        big_id_dtype : type
            dtype for large pandas.Series of integer ID values.

        small_id_dtype : type
            dtype for small pandas.Series of integer ID values.

        list_position_base : int
            Index value to be assigned to first element in items with
            list delimiters.

        Returns
        -------
        dict
            Has the following elements

            'strings': pandas.DataFrame with dtypes
                {'string': str, 'node_type_id': small_id_dtype}.
                Index type is big_id_dtype

            'links': pandas.DataFrame with dtypes
                {'src_string_id': big_id_dtype,
                 'tgt_string_id': big_id_dtype,
                 'ref_string_id': big_id_dtype,
                 'link_type_id: small_id_dtype}
                Index type is big_id_dtype

            'node_types': pandas.Series with dtype str. Index type is
                small_id_dtype.

            'link_types': pandas.Series with dtype str. Index type is
                small_id_dtype.

            'entry_prefixes': pandas.Series with dtype str. Index type
                is small_id_dtype.

            'item_labels': pandas.Series with dtype str. Index type is
                small_id_dtype.
        '''
        data = pd.read_csv(
            filepath_or_buffer,
            skiprows=skiprows,
            skipinitialspace=True
        )

        # see shnd.util.set_string_dtype docstring for comment on new
        # pandas string dtype
        # shnd_input = shnd.util.set_string_dtype(shnd_input)

        data = _normalize_shorthand(
            data,
            comment_char,
            fill_cols,
            drop_na
        )

        # Get any metadata for links between entries
        link_metadata = data.loc[
            data['link_tags_or_override'].notna(),
            'link_tags_or_override'
        ]

        # replace text column labels with integers so we compute on
        # integer indexes
        csv_column_id_map = _create_id_map(
            list(data.columns),
            dtype=pd.UInt8Dtype()
        )
        data.columns = csv_column_id_map.array

        # Get input columns with entry strings
        data = data.loc[
            :,
            csv_column_id_map[['left_entry', 'right_entry', 'reference']]
        ]

        # Stack the entries into a pandas.Series. Values are entry
        # strings and the index is a multiindex with the csv row and
        # column for each entry
        data = data.stack().dropna()

        # Check for duplicate entry strings
        entry_is_duplicated = data.duplicated()

        if entry_is_duplicated.any():
            # If there are duplicate entries, we want to cache their
            # index values and drop the string values so we don't do
            # unnecessary work on the strings
            dplct_entries = data.loc[entry_is_duplicated, :]
            data = data.loc[~entry_is_duplicated, :]

            # Make a map between distinct entry strings and their
            # csv index values
            distinct_string_map = pd.Series(
                data.index.to_flat_index(),
                index=data
            )

            # Map the duplicate strings to csv index values and convert
            # the result into a dataframe. The dataframe has a
            # multiindex with the csv row and column of the duplicate
            # strings. The dataframe columns contain the csv rows and
            # columns of the distinct string value for each duplicate
            # string
            dplct_entries = dplct_entries.map(distinct_string_map)
            dplct_entries = pd.DataFrame(
                tuple(dplct_entries.array),
                columns=['string_csv_row', 'string_csv_col'],
                index=dplct_entries.index
            )

            # Reset the index and rename columns. "string" csv indexes
            # refer to the location of a distinct string value. "entry"
            # csv indexes refer to the location of a shorthand# entry
            # whose string value is potentially a duplicate.
            dplct_entries = dplct_entries.reset_index()
            dplct_entries = dplct_entries.rename(
                columns={
                    'level_0': 'entry_csv_row',
                    'level_1': 'entry_csv_col'
                }
            )
            dplct_entries = dplct_entries.astype({
                'entry_csv_row': big_id_dtype,
                'entry_csv_col': pd.UInt8Dtype(),
                'string_csv_row': big_id_dtype,
                'string_csv_col': pd.UInt8Dtype()
            })

        # Remove clearspace around the entry strings
        data = data.str.strip()

        '''*********************************************
        data is currently a string-valued Series
        *********************************************'''

        # Read the entry syntax
        entry_syntax = shnd.syntax_parsing.validate_entry_syntax(
            self.entry_syntax,
            case_sensitive=self.syntax_case_sensitive
        )

        # Parse entries in the input text
        data = shnd.entry_parsing.parse_entries(
            data,
            entry_syntax,
            self.item_separator,
            self.default_entry_prefix,
            self.space_char,
            self.na_string_values
        )
        data = data.reset_index()
        data = data.rename(
            columns={
                'level_0': 'csv_row',
                'level_1': 'csv_col',
                'grp_prefix': 'entry_prefix',
                'level_3': 'item_label'
            }
        )
        # replace missing entry prefixes with default value
        prefix_isna = data['entry_prefix'].isna()
        data.loc[prefix_isna, 'entry_prefix'] = self.default_entry_prefix

        # For any strings that represent null values, overwrite the node
        # type inferred from the syntax with the null node type
        null_strings = data['string'].isin(self.na_string_values)
        data.loc[null_strings, 'node_type'] = self.na_node_type

        dtypes = {
            'csv_row': big_id_dtype,
            'csv_col': pd.UInt8Dtype(),
            # 'entry_prefix': pd.StringDtype(),
            # 'item_label': pd.StringDtype(),
            # 'string': pd.StringDtype(),
            # 'node_type': pd.StringDtype(),
            # 'link_type': pd.StringDtype(),
            # 'node_type': pd.StringDtype()
        }
        # can't use pd.StringDtype() throughout because it currently
        # doesn't allow construction with null types other than pd.NA.
        # This will likely change soon
        # https://github.com/pandas-dev/pandas/pull/41412

        data = data.astype(dtypes)
        data.index = data.index.astype(big_id_dtype)

        '''******************************************************
        data is currently a DataFrame with these columns:
            ['csv_row', 'csv_col', 'entry_prefix', 'item_label',
             'string', 'node_type', 'link_type', 'node_tags']
        ******************************************************'''

        # Map string-valued entry prefixes to integer IDs
        entry_prefix_id_map = _create_id_map(
            data['entry_prefix'],
            dtype=small_id_dtype
        )
        # Replace entry prefixes in the mutable data with integer IDs
        data['entry_prefix'] = data['entry_prefix'].map(
            entry_prefix_id_map
        )
        data = data.rename(
            columns={'entry_prefix': 'entry_prefix_id'}
        )

        # Map string-valued item labels to integer IDs
        item_label_id_map = _create_id_map(
            data['item_label'],
            dtype=small_id_dtype
        )
        # Replace item labels in the mutable data with integer IDs
        data['item_label'] = data['item_label'].map(
            item_label_id_map
        )
        data = data.rename(
            columns={'item_label': 'item_label_id'}
        )

        # These link types are required to complete linking operations
        # later
        link_types = pd.Series(['entry', 'tagged', 'requires'])

        # Map string-valued link types to integer IDs
        link_types = _create_id_map(
            pd.concat([link_types, data['link_type']]),
            dtype=small_id_dtype
        )
        # Replace link types in the mutable data with integer IDs
        data['link_type'] = data['link_type'].map(
            link_types
        )
        data = data.rename(
            columns={'link_type': 'link_type_id'}
        )
        # Mutate link_types into a series whose index is integer IDs and
        # whose values are string-valued link types
        link_types = pd.Series(link_types.index, index=link_types)

        '''******************************************************
        data is currently a DataFrame with these columns:
            ['csv_row', 'csv_col', 'entry_prefix_id', 'item_label_id',
             'string', 'node_type', 'link_type_id', 'node_tags']
        ******************************************************'''

        # Split items that have list delimiters in the entry syntax
        data = data.groupby(
            by=['entry_prefix_id', 'item_label_id'],
            dropna=False
        )
        data = data.apply(
            _expand_items,
            entry_syntax,
            entry_prefix_id_map,
            item_label_id_map
        )

        # Locate items that do not have a list delimiter
        item_delimited = data['string'].map(
            pd.api.types.is_list_like
        )
        item_not_delimited = data.loc[~item_delimited].index

        # Explode the delimited strings
        data = data.explode('string')

        # Make a copy of the index, drop values that do not refer to
        # delimited items, then groupby the remaining index values and
        # do a cumulative count to get the position of each element in
        # each item that has a list delimiter
        itm_list_pos = data.index
        itm_list_pos = pd.Series(
            itm_list_pos.array,
            index=itm_list_pos,
            dtype=small_id_dtype
        )

        itm_list_pos.loc[itm_list_pos.isin(item_not_delimited)] = pd.NA
        itm_list_pos = itm_list_pos.dropna().groupby(itm_list_pos.dropna())
        itm_list_pos = itm_list_pos.cumcount()
        # Shift the values by the list position base
        itm_list_pos = itm_list_pos + list_position_base

        itm_list_pos = itm_list_pos.astype(small_id_dtype)

        # Store the item list positions and reset the index
        itm_list_pos = itm_list_pos.array
        data.loc[item_delimited, 'item_list_position'] = itm_list_pos
        data = data.reset_index(drop=True)

        '''**********************************************************
        data is currently a DataFrame with these columns:
            ['csv_row', 'csv_col', 'entry_prefix_id', 'item_label_id',
             'string', 'node_type', 'link_type_id', 'node_tags',
             'item_list_position']

        Done processing entry strings.
        Replace string values in data with integer ID values.
        **********************************************************'''

        # The strings dataframe is a relation between a string value
        # and a node type. Its index is integer string IDs
        strings = data[['string', 'node_type']].drop_duplicates(
            subset='string'
        )
        strings = strings.reset_index(drop=True)
        strings.index = strings.index.astype(big_id_dtype)

        # Drop node types from the mutable data
        data = data.drop('node_type', axis='columns')

        # Replace strings in the mutable data with integer IDs
        data['string'] = data['string'].map(
            pd.Series(strings.index, index=strings['string'])
        )
        data = data.rename(columns={'string': 'string_id'})

        # These node types will be required later
        node_types = pd.Series([
            'shorthand_text',
            'shorthand_entry_syntax',
            'shorthand_link_syntax',
            'python_function'
        ])

        # Map string-valued node types to integer IDs
        node_types = _create_id_map(
            pd.concat([node_types, strings['node_type']]),
            dtype=small_id_dtype
        )

        # Replace string-valued node types in the strings dataframe with
        # integer IDs
        strings['node_type'] = strings['node_type'].map(node_types)
        strings = strings.rename(columns={'node_type': 'node_type_id'})
        strings = strings.astype({
            'string': str,
            'node_type_id': small_id_dtype
        })

        # Mutate node_types into a series whose index is integer IDs and
        # whose values are string-valued node types
        node_types = pd.Series(node_types.index, index=node_types)

        '''**********************************************************
        data is currently a DataFrame with these columns:
            ['csv_row', 'csv_col', 'entry_prefix_id', 'item_label_id',
             'string_id', 'link_type_id', 'node_tags',
             'item_list_position']

        id columns and item_list_position are integer-valued.

        Generate links defined by the entry and link syntax
        **********************************************************'''

        columns_required_for_links = [
            'csv_row', 'csv_col', 'string_id', 'link_type_id'
        ]

        # If the entry syntax indicated that there should be links
        # between an item and the string that contains it, get the
        # string ID of the entry
        links = data[columns_required_for_links].groupby(
            by=['csv_row', 'csv_col'],
            group_keys=False
        )
        links = links.apply(_get_item_link_source_IDs)
        links.index = data.index.copy()

        # A shorthand link is a relation between four entities
        # represented by integer IDs:
        #
        # src_string_id (string representing the source end of the link)
        # tgt_string_id (string representing the target end of the link)
        # ref_string_id (string representing the context of the link)
        # link_type_id (the link type)
        #
        # To generate these we first locate items with links to their
        # own entry strings and then treat the entry strings as source
        # strings.
        has_link = ~data['link_type_id'].isna()

        # The reference string for links between items and the entry
        # containing them is the full text of the input file, which is
        # not in the current data set, so set reference strings to null.
        links = pd.DataFrame({
            'src_string_id': links.loc[has_link].array,
            'ref_string_id': pd.NA
        })

        # Get the target string, link type, and list position from the
        # data set
        data_cols = ['string_id', 'link_type_id', 'item_list_position']
        links = pd.concat(
            [links, data.loc[has_link, data_cols].reset_index(drop=True)],
            axis='columns'
        )
        links = links.rename(columns={'string_id': 'tgt_string_id'})

        # Every entry string in a shorthand text is also the target of a
        # link of type 'entry' whose source is the full text of the
        # input file. This allows direct selection of the original entry
        # strings. The text of the input file is not in the current
        # data set, so set the source strings to null.
        tgt_string_ids = data.loc[data['item_label_id'].isna(), 'string_id']
        entry_links = pd.DataFrame({
            'src_string_id': pd.NA,
            'tgt_string_id': tgt_string_ids.drop_duplicates().array,
            'ref_string_id': pd.NA,
            'link_type_id': link_types.loc[link_types == 'entry'].index[0],
            'item_list_position': pd.NA
        })
        links = pd.concat([links, entry_links])

        # Get entry prefixes for each side of links defined in the link
        # syntax
        left_prefixes = _get_entry_prefix_ids(
            'left',
            data,
            dplct_entries,
            csv_column_id_map
        )
        right_prefixes = _get_entry_prefix_ids(
            'right',
            data,
            dplct_entries,
            csv_column_id_map
        )

        # Recover entry prefix IDs for duplicate entries whose original
        # string value is in a different csv column
        _copy_cross_duplicates(left_prefixes, right_prefixes)

        # Done with the "entry" csv columns, so drop them
        left_prefixes = left_prefixes.drop('entry_csv_col', axis='columns')
        right_prefixes = right_prefixes.drop('entry_csv_col', axis='columns')

        left_prefixes = left_prefixes.rename(
            columns={'string_csv_row': 'L_str_csv_row',
                     'string_csv_col': 'L_str_csv_col'}
        )

        right_prefixes = right_prefixes.rename(
            columns={'string_csv_row': 'R_str_csv_row',
                     'string_csv_col': 'R_str_csv_col'}
        )

        # Pair up the prefixes so we can generate links from the link
        # syntax
        prefix_pairs = left_prefixes.merge(right_prefixes)

        link_types, link_syntax = shnd.syntax_parsing.parse_link_syntax(
            self.link_syntax,
            self.entry_syntax,
            entry_prefix_id_map,
            link_types,
            item_label_id_map,
            case_sensitive=self.syntax_case_sensitive
        )

        # Get string IDs for links whose sources and targets are matched
        # one-to-one according to the link syntax
        link_has_no_list = link_syntax['list_mode'].isna()
        link_is_one_to_one = (link_syntax['list_mode'] == '1:1')
        list_mode_subset = link_has_no_list | link_is_one_to_one

        sources = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'src_',
            subset=list_mode_subset,
            columns=['entry_csv_row', 'string_id', 'link_type_id']
        )
        targets = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'tgt_',
            subset=list_mode_subset,
            columns=['string_id', 'item_list_position']
        )
        references = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'ref_',
            subset=list_mode_subset,
            columns=['entry_csv_row', 'string_id']
        )

        one_to_one_links = pd.concat([sources, targets], axis='columns')
        one_to_one_links = one_to_one_links.merge(
            references,
            on='entry_csv_row',
            how='left'
        )

        # Get string IDs for links whose sources and targets are not
        # matched one-to-one
        list_mode_subset = link_syntax['list_mode'].isin(['1:m', 'm:1', 'm:m'])

        sources = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'src_',
            subset=list_mode_subset,
            columns=['entry_csv_row', 'string_id', 'link_type_id']
        )
        targets = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'tgt_',
            subset=list_mode_subset,
            columns=['entry_csv_row', 'string_id', 'item_list_position']
        )
        references = _get_link_component_string_ids(
            prefix_pairs,
            data,
            link_syntax,
            'ref_',
            subset=list_mode_subset,
            columns=['entry_csv_row', 'string_id']
        )

        other_links = sources.merge(targets, on='entry_csv_row')
        other_links = other_links.merge(references, on='entry_csv_row')

        one_to_one_links = shnd.util.normalize_types(
            one_to_one_links,
            links,
            strict=False
        )
        other_links = shnd.util.normalize_types(
            other_links,
            links,
            strict=False
        )
        links = pd.concat([links, one_to_one_links, other_links])

        links = links.reset_index(drop=True)
        links = links.rename(columns={'item_list_position': 'list_position'})
        links = links.astype({
            'src_string_id': big_id_dtype,
            'tgt_string_id': big_id_dtype,
            'ref_string_id': big_id_dtype,
            'link_type_id': small_id_dtype,
            'entry_csv_row': big_id_dtype,
            'list_position': small_id_dtype
        })

        # Link metadata, overriding link types created above and/or
        # adding tags to links, was extracted from entries near the
        # begining of this function. Now process the link types and
        # insert the tag strings into the links frame.

        # Escape any regex metacharacters in the item separator so we
        # can use it in regular expressions
        regex_item_separator = shnd.util.escape_regex_metachars(
            self.item_separator
        )

        # Extract the link type overrides from the link metadata with a
        # regular expression
        # TAKES ONLY THE FIRST MATCH, OTHERS CONSIDERED TAGS
        link_type_regex = rf"^(?:.*?)(lt{regex_item_separator}\S+)"
        link_type_overrides = link_metadata.str.extract(link_type_regex)
        link_type_overrides = link_type_overrides.stack().dropna()
        link_type_overrides.index = link_type_overrides.index.droplevel(1)
        link_type_overrides = link_type_overrides.str.split(
            self.item_separator,
            expand=True
        )

        try:
            # If we found any new link types, process them
            link_type_overrides = link_type_overrides[1]

            # Add overridden types to the link types series
            new_link_types = link_type_overrides.loc[
                ~link_type_overrides.isin(link_types)
            ]
            link_types = pd.concat([
                link_types,
                shnd.util.normalize_types(new_link_types, link_types)
            ])

            # Map string-valued type overrides to integer link type IDs
            link_type_overrides = link_type_overrides.map(
                pd.Series(link_types.index, index=link_types.array)
            )

            # Replace overriden link type IDs
            link_type_overrides = links['entry_csv_row'].dropna().map(
                link_type_overrides
            )
            links['link_type_id'].update(link_type_overrides.dropna())

        except KeyError:
            # If there wasn't a second column in link_type_overrides
            # then there are no new links to process.
            pass

        # The link metadata index currently refers to csv rows. Replace
        # it with integer link index values.
        row_map = links.query('entry_csv_row.isin(@link_metadata.index)')
        row_map = pd.Series(
            row_map.index,
            index=row_map['entry_csv_row'].array
        )
        link_metadata.index = link_metadata.index.map(row_map)

        # We're done with the csv information
        links = links.drop('entry_csv_row', axis='columns')

        '''***********************************************************
        Done with links generated from syntax definitions.

        Extract tag strings and create links between strings and tags.
        ***********************************************************'''

        # Add node type for tags if it doesn't already exist
        if 'tag' not in node_types:
            index = pd.Series(
                node_types.index.max() + 1,
                dtype=node_types.index.dtype
            )
            new_type = pd.Series('tag', index=index)
            node_types = pd.concat([node_types, new_type])

        # Cache the node type ID for tag strings
        tag_node_type_id = node_types.loc[node_types == 'tag'].index[0]

        # Add link type for tags if it doesn't already exist
        '''if 'tagged' not in link_types:
            index = pd.Series(
                link_types.index.max() + 1,
                dtype=link_types.index.dtype
            )
            new_type = pd.Series('tagged', index=index)
            link_types = pd.concat([link_types, new_type])'''

        # Cache the type ID for tag links
        tagged_link_type = link_types.loc[link_types == 'tagged'].index[0]

        # PROCESS NODE TAGS

        # Make a relation between node tag strings and ID values for the
        # tagged strings
        tags = data.loc[
            data['node_tags'].notna(),
            ['string_id', 'node_tags']
        ]
        tags = pd.Series(tags['node_tags'].array, index=tags['string_id'])
        tags = tags.str.split().explode()

        # tags is now a pandas.Series whose index is string IDs and
        # whose values are individual tag strings

        # Tag strings should be added to the strings frame if they are
        # not present in the strings frame OR if they are present and
        # the existing string has a node type that isn't the tag node
        # type
        new_strings = shnd.util.get_new_typed_values(
            tags,
            strings,
            'string',
            'node_type_id',
            tag_node_type_id
        )
        new_strings = pd.DataFrame(
            {'string': new_strings.array, 'node_type_id': tag_node_type_id},
            index=tags.array
        )
        new_strings = shnd.util.normalize_types(new_strings, strings)

        strings = pd.concat([strings, new_strings])

        # convert the tag strings to string ID values
        tags = tags.map(
            pd.Series(new_strings.index, index=new_strings['string'])
        )

        # Add links for the tags. Reference string IDs are null because
        # the reference string for a tag is the full text of the input
        # file, which is inserted by the caller.
        new_links = pd.DataFrame({
            'src_string_id': tags.index,
            'tgt_string_id': tags.array,
            'ref_string_id': pd.NA,
            'link_type_id': tagged_link_type
        })
        new_links = shnd.util.normalize_types(new_links, links)

        links = pd.concat([links, new_links])

        # PROCESS LINK TAGS

        # Extract the link tags from the link metadata with a regular
        # expression
        link_tag_regex = rf"lt{regex_item_separator}\S+"
        tags = link_metadata.str.replace(link_tag_regex, '', n=1, regex=True)
        tags = tags.loc[tags != '']

        # Convert any one-to-one link target list positions into strings
        # and append them to the tag strings
        list_pos = links['list_position'].dropna().astype(str)
        has_list_pos_and_tags = list_pos.index.intersection(tags.index)
        list_pos_but_no_tags = list_pos.index.difference(tags.index)

        if has_list_pos_and_tags.empty:
            tags = pd.concat([tags, list_pos])

        elif list_pos.difference(has_list_pos_and_tags).empty:
            tags.loc[list_pos.index] = [
                ' '.join(pair) for pair in
                zip(tags.loc[list_pos.index], list_pos)
            ]

        else:
            pairs = zip(
                tags.loc[has_list_pos_and_tags],
                list_pos.loc[has_list_pos_and_tags]
            )
            tags.loc[has_list_pos_and_tags] = [
                ' '.join(pair) for pair in pairs
            ]

            tags = pd.concat([tags, list_pos.loc[list_pos_but_no_tags]])

        # Convert the space-delimited tag strings to lists of strings
        tags = tags.str.split().explode()

        # Add the tag strings to the rest of the strings
        new_strings = tags.drop_duplicates()
        # new_strings = new_strings.loc[~new_strings.isin(strings['string'])]
        new_strings = shnd.util.get_new_typed_values(
            new_strings,
            strings,
            'string',
            'node_type_id',
            tag_node_type_id
        )
        new_strings = pd.DataFrame(
            {'string': new_strings.array, 'node_type_id': tag_node_type_id}
        )
        new_strings = shnd.util.normalize_types(new_strings, strings)

        strings = pd.concat([strings, new_strings])

        # convert the tag strings to string ID values
        tag_strings = strings.loc[strings['node_type_id'] == tag_node_type_id]
        tags = tags.map(
            pd.Series(tag_strings.index, index=tag_strings['string'])
        )

        # Relations between links and string-valued tags stored in the
        # strings frame aren't representable as links in the links
        # frame, so make a many-to-many frame relating link ID values to
        # string IDs
        link_tags = pd.DataFrame({
            'link_id': tags.index,
            'tag_string_id': tags.array
        })

        '''************
        Done with tags.
        ************'''

        # Mutate entry prefix and item label maps into string-valued
        # series with integer indexes
        entry_prefix_id_map = pd.Series(
            entry_prefix_id_map.index,
            index=entry_prefix_id_map
        )
        item_label_id_map = pd.Series(
            item_label_id_map.index,
            index=item_label_id_map
        )

        return shnd.ParsedShorthand(
            strings=strings,
            links=links,
            link_tags=link_tags,
            node_types=node_types,
            link_types=link_types,
            entry_prefixes=entry_prefix_id_map,
            item_labels=item_label_id_map,
            item_separator=self.item_separator,
            default_entry_prefix=self.default_entry_prefix,
            space_char=self.space_char,
            comment_char=comment_char,
            na_string_values=self.na_string_values,
            na_node_type=self.na_node_type,
            syntax_case_sensitive=self.syntax_case_sensitive
        )

    def parse_text(
        self,
        filepath_or_buffer,
        skiprows,
        comment_char,
        fill_cols='left_entry',
        drop_na='right_entry',
        big_id_dtype=pd.Int32Dtype(),
        small_id_dtype=pd.Int8Dtype(),
        list_position_base=1
    ):
        ####################
        # Validate arguments
        ####################

        # We need to use the contents of the input file for parsing but
        # we also need to access the text after parsing. If we're passed
        # a buffer then it will be empty after it's passed to
        # pandas.read_csv, so we write it to a temp file.
        if 'read' in dir(filepath_or_buffer):
            if not Path.is_file('temp.shorthand'):
                with open('temp.shorthand', 'w') as f:
                    f.write(filepath_or_buffer)

            else:
                raise RuntimeError(
                    'Cannot overwrite file temp.shorthand in the '
                    'current working directory. Delete, move, or '
                    'rename and try again.'
                )

            filepath_or_buffer = 'temp.shorthand'

        # Hash the input text so we can validate it when we read it back
        with open(filepath_or_buffer, 'r') as f:
            input_hash = hash(f.read())

        comment_char = str(comment_char)
        if len(comment_char) > 1:
            raise ValueError('comment_char must be a single character')

        if not shnd.util.iterable_not_string(fill_cols):
            if fill_cols is None:
                fill_cols = []

            fill_cols = [fill_cols]

        if not shnd.util.iterable_not_string(drop_na):
            if drop_na is None:
                drop_na = []

            drop_na = [drop_na]

        if not pd.api.types.is_integer_dtype(big_id_dtype):
            raise ValueError(
                'big_id_dtype must be an integer type recognized by '
                'pandas.api.types.is_integer_dtype'
            )

        if not pd.api.types.is_integer_dtype(small_id_dtype):
            raise ValueError(
                'small_id_dtype must be an integer type recognized by '
                'pandas.api.types.is_integer_dtype'
            )

        list_position_base = int(list_position_base)

        ###########################
        # Done validating arguments
        ###########################

        # Parse input text
        parsed = self._apply_syntax(
            filepath_or_buffer,
            skiprows,
            comment_char,
            fill_cols,
            drop_na,
            big_id_dtype,
            small_id_dtype,
            list_position_base
        )

        # Read input text from temp file
        with open(filepath_or_buffer, 'r') as f:
            new_string = f.read()

            if hash(new_string) != input_hash:
                raise RuntimeError('input text was modified during parsing')

            # Delete the temp file if it exists
            Path.unlink(Path('temp.shorthand'), missing_ok=True)

            # Build an input_text row for the strings frame
            txt_node_type_id = parsed.id_lookup('node_types', 'shorthand_text')
            new_string = {
                'string': new_string,
                'node_type_id': txt_node_type_id
            }
            new_string = shnd.util.normalize_types(
                new_string,
                parsed.strings
            )

            # Insert a strings row for the input text
            parsed.strings = pd.concat([parsed.strings, new_string])

        input_text_string_id = parsed.strings.index[-1]

        # Insert a strings row for the current function
        func_node_type_id = parsed.id_lookup('node_types', 'python_function')
        new_string = {
            'string': 'Shorthand.parse_text',
            'node_type_id': func_node_type_id
        }
        new_string = shnd.util.normalize_types(new_string, parsed.strings)
        parsed.strings = pd.concat([parsed.strings, new_string])

        parse_function_string_id = parsed.strings.index[-1]

        # Insert a strings row for the entry syntax
        entry_syntax_node_type_id = parsed.id_lookup(
            'node_types',
            'shorthand_entry_syntax'
        )
        new_string = {
            'string': self.entry_syntax,
            'node_type_id': entry_syntax_node_type_id
        }
        new_string = shnd.util.normalize_types(new_string, parsed.strings)
        parsed.strings = pd.concat([parsed.strings, new_string])

        # Insert a link between the input text and the entry syntax
        new_link = {
            'src_string_id': input_text_string_id,
            'tgt_string_id': new_string.index[0],
            'ref_string_id': parse_function_string_id,
            'link_type_id': parsed.id_lookup('link_types', 'requires')
        }
        new_link = shnd.util.normalize_types(new_link, parsed.links)
        parsed.links = pd.concat([parsed.links, new_link])

        try:
            assert self.link_syntax
            # If we were given a link syntax, create a row for it in the
            # strings frame and create a link between the input text
            # and the link syntax
            link_syntax_node_type_id = parsed.id_lookup(
                'node_types',
                'shorthand_link_syntax'
            )
            new_string = {
                'string': self.link_syntax,
                'node_type_id': link_syntax_node_type_id
            }
            new_string = shnd.util.normalize_types(
                new_string,
                parsed.strings
            )
            parsed.strings = pd.concat(
                [parsed.strings, new_string]
            )

            new_link = {
                'src_string_id': input_text_string_id,
                'tgt_string_id': new_string.index[0],
                'ref_string_id': parse_function_string_id,
                'link_type_id': parsed.id_lookup('link_types', 'requires')
            }
            new_link = shnd.util.normalize_types(new_link, parsed.links)
            parsed.links = pd.concat([parsed.links, new_link])

        except AttributeError:
            pass

        # Links whose reference string ID is missing should have the
        # input file as their reference string
        ref_isna = parsed.links['ref_string_id'].isna()
        parsed.links.loc[ref_isna, 'ref_string_id'] = input_text_string_id

        # Links whose source string ID is missing should have the input
        # file as their source string
        src_isna = parsed.links['src_string_id'].isna()
        parsed.links.loc[src_isna, 'src_string_id'] = input_text_string_id

        return parsed
