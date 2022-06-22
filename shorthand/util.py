import pandas as pd


def collapse_columns(df, columns):

    if iterable_not_string(columns):
        if len(columns) > 1:
            if not df[columns].count(axis=1).lt(2).all():
                raise ValueError(
                    'found multiple values in a single row for columns '
                    '{}'.format(columns)
                )
            return df[columns].ffill(axis=1).dropna(axis=1)

        elif len(columns) == 1:
            return pd.DataFrame(df[columns])

        else:
            return pd.DataFrame(dtype='object')

    elif (type(columns) == str):
        return df[columns]

    else:
        raise ValueError('unrecognized input value for columns')


def create_id_map(domain, drop_na=True, **kwargs):
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


def map_values_to_id(df, label, dtype=pd.UInt64Dtype()):
    '''
    THIS FUNCTION MUTATES ITS FIRST ARGUMENT

    Make a map from values in a dataframe column to integers, replace
    the column with integer values, and return a pandas.Series with the
    integers as an index and the original column values as the array.
    '''

    id_map = create_id_map(df[label], dtype=dtype)
    df[label] = df[label].map(id_map)

    return pd.Series(id_map.index, index=id_map)


def escape_regex_metachars(s):
    s = s.replace("\\", "\\\\")
    metachars = '.^$*+?{}[]|()'
    for metachar in [c for c in metachars if c in s]:
        s = s.replace(metachar, f'\\{metachar}')
    return s


def extend_id_map(domain,
                  existing_domain,
                  existing_range=None,
                  drop_na=True,
                  **kwargs):
    '''
    Map distinct values in a domain which are not in an existing domain
    to integers that do not overlap with an existing range. Additional
    keyword arguments are passed to the pandas.Series constructor when
    the map series is created.

    Parameters
    ----------
    domain : list-like (coercible to pandas.Series)
        Arbitrary set of values to map. May contain duplicates.

    existing_domain : list-like
        Set of values already present in a map from values to integers.

    existing_range : list-like or None, default None
        Range of integers already present in the range of a map. If
        None, assume existing_domain.index contains the existing range.

    drop_na : bool, default True
        Ignore null values and map only non-null values to integers.

    Examples
    --------
    >>> existing_domain = pd.Series(['a', 'a', 'b', 'f', 'b'])
    >>> new_domain = ['a', 'b', 'z', pd.NA]
    >>> _extend_id_map(new_domain,
                       existing_domain,
                       dtype=pd.UInt16Dtype())

    z    5
    dtype: UInt16

    >>> _extend_id_map(new_domain,
                       existing_domain,
                       dtype=pd.UInt16Dtype(),
                       drop_na=False)

    z       5
    <NA>    6
    dtype: UInt16
    '''
    # check if domain object has autocorr and loc attributes like a
    # pandas.Series and convert if not
    try:
        assert domain.autocorr
        assert domain.loc
        # make a copy so we can mutate one (potentially large) object
        # instead of creating additional references
        domain = domain.copy()
    except AttributeError:
        domain = pd.Series(domain)

    if drop_na:
        domain = domain.loc[~domain.isna()]

    domain_is_new = ~domain.isin(existing_domain)

    if domain_is_new.any():
        domain = domain.loc[domain_is_new].drop_duplicates()

        if existing_range is None:
            new_ids = non_intersecting_sequence(
                len(domain),
                existing_domain.index
            )
        else:
            new_ids = non_intersecting_sequence(
                len(domain),
                existing_range
            )

    else:
        domain = []
        new_ids = []

    return pd.Series(new_ids, index=domain, **kwargs)


def get_single_value(
    df_or_s,
    label,
    none_ok=False,
    group_key=None
):

    try:
        # If this assertion passes then assume input is a pandas.Series
        assert df_or_s.str

        if df_or_s.empty:
            raise ValueError(f'Series with label "{label}" is empty.')
        elif len(df_or_s) == 1:
            return df_or_s.array[0]

        values = df_or_s.value_counts()

        num_values = len(values)

        if num_values == 1:
            return values.index[0]
        elif num_values > 1:
            raise ValueError(f'Found multiple values with label "{label}"')
        elif num_values < 1:
            if not none_ok:
                raise ValueError(f'Found only nan values with label "{label}"')
            else:
                return None

    except AttributeError:

        # If the assertion failed then assume input is a pandas.DataFrame
        if len(df_or_s[label]) == 1:
            return df_or_s[label].array[0]

        if group_key is None:
            message_tail = f'in column "{label}"'

        else:
            message_tail = 'in column "{}", group key "{}"'.format(
                label,
                group_key
            )

        values = df_or_s[label].value_counts()

        num_values = len(values)

        if num_values == 1:
            return values.index[0]
        elif num_values > 1:
            raise ValueError('Found multiple values {}'.format(message_tail))
        elif num_values < 1:
            if not none_ok:
                raise ValueError('No values found {}'.format(message_tail))
            else:
                return None


def iterable_not_string(x):
    '''
    Check if input has an __iter__ method and then determine if it's a
    string by checking for a casefold method.
    '''
    try:
        assert x.__iter__

        try:
            assert x.casefold
            return False

        except AttributeError:
            return True

    except AttributeError:
        return False


def normalize_types(to_norm, template, strict=True, continue_idx=True):
    '''
    Create an object from to_norm that can be concatenated with template
    such that the concatenated object and its index will have the same
    dtypes as the template.

    If template is pandas.DataFrame and to_norm is dict-like or
    list-like, convert each element of to_norm to a pandas.Series with
    dtypes that conform to the dtypes of a template dataframe and and
    concatenate them together as columns in a dataframe.

    If template is pandas.Series, convert to_norm to a Series with the
    appropriate array dtype or, if to_norm is dict-like,
    convert its values to an array with the appropriate dtype and if
    both to_norm and template have numeric indexes, also normalize the
    index dtype. If to_norm is dict-like and either to_norm or template
    does not have a numeric index, use to_norm.keys() as the output
    index.

    Parameters
    ----------
    to_norm : dict-like or list-like
        A set of objects to treat as columns of an output
        pandas.DataFrame and whose types will be coerced.

    template : pandas.DataFrame or pandas.Series
        Output dtypes will conform to template.dtypes and the output
        index dtype will be template.index.dtype

    strict : bool, default True
        If True and to_norm is dict-like, only objects whose keys are
        column labels in the template will be included as columns in the
        output dataframe.

        If True, to_norm is list-like, and template has N columns,
        include only the first N elements of to_norm in the output
        dataframe.

        If False and to_norm is dict-like, normalize dtypes for objects
        whose keys are column labels in the template and include the
        other elements of to_norm as columns with dtypes inferred by
        the pandas.Series constructor.

        If False, to_norm is list-like, and template has N columns,
        normalize dtypes for the first N elements of to_norm and include
        the other elements of to_norm as columns with dtypes inferred by
        the pandas.Series constructor. Labels for the extra columns in
        the output dataframe will be integers counting from N.

    continue_index : bool, default True
        If True and template has a numerical index, the index of the
        returned object will be a sequence of integers which fills
        gaps in and/or extends to_norm.index

    Returns
    -------
    pandas.DataFrame
        Has as many rows as the longest element of to_norm.

        If to_norm is dict-like and strict is True, output includes
        only objects in to_norm whose keys are also column labels in the
        template.

        If to_norm is list-like and strict is True, output has the same
        width as the template.

        If strict is False, output has one column for each element in
        to_norm.
    '''

    # get the size of to_norm

    try:
        # to_norm is treated as columns, so if it has a columns
        # attribute, we want the size of that instead of the length
        num_elements = len(to_norm.columns)
    except AttributeError:
        num_elements = len(to_norm)

    # check if the template index has a numeric dtype
    templt_idx_is_nmrc = pd.api.types.is_numeric_dtype(template.index)

    try:
        tmplt_columns = template.columns
        # If the template is a dataframe, get its width.
        num_tmplt_columns = len(tmplt_columns)

    except AttributeError:
        # Template is not dataframe-like.
        # Treat it as 1D object with attributes dtype and index

        try:
            assert to_norm.items
            # to_norm is dict-like, so process its keys as an index

            norm_keys_are_nmrc = pd.api.types.is_numeric_dtype(
                pd.Series(to_norm.keys())
            )
            if strict and norm_keys_are_nmrc and templt_idx_is_nmrc:
                index = non_intersecting_sequence(
                    to_norm.keys(),
                    template.index
                )
            else:
                index = to_norm.keys()

            try:
                # values is callable for a dict-like but not a
                # pandas.Series
                values = to_norm.values()

            except TypeError:
                values = to_norm.array

        except AttributeError:
            if strict and templt_idx_is_nmrc:
                index = non_intersecting_sequence(num_elements, template.index)
            else:
                index = range(num_elements)

            values = to_norm

        index = pd.Index(index, dtype=template.index.dtype)
        return pd.Series(values, index=index, dtype=template.dtype)

    try:
        # check if to_norm is dict-like
        assert to_norm.items
        # if so, optionally get extra columns
        if (num_elements > num_tmplt_columns) and not strict:
            extra_columns = {
                k: v for k, v in to_norm.items() if k not in tmplt_columns
            }

    except AttributeError:
        # to_norm is not dict-like
        if num_elements < num_tmplt_columns:
            raise ValueError(
                'If to_norm is list-like, to_norm must have at least '
                'as many elements as there are columns in template.'
            )
        # optionally get extra columns
        if (num_elements > num_tmplt_columns) and not strict:
            extra_columns = dict(zip(
                range(num_tmplt_columns, num_elements),
                to_norm[num_tmplt_columns:]
            ))
        # make the list-like dict-like
        to_norm = dict(zip(tmplt_columns, to_norm[:num_tmplt_columns]))

    to_norm = [
        pd.Series(v, dtype=template[k].dtype, name=k)
        for k, v in to_norm.items() if k in tmplt_columns
    ]

    try:
        to_norm += [pd.Series(v, name=k) for k, v in extra_columns.items()]
    except NameError:
        pass

    new_df = pd.concat(to_norm, axis='columns')

    if continue_idx and templt_idx_is_nmrc:
        template_idx_is_sequential = pd.Series(template.index).diff().iloc[1:]
        template_idx_is_sequential = (template_idx_is_sequential == 1).all()

        if template_idx_is_sequential & template.index.is_monotonic:
            new_index_min = template.index.max() + 1
            index = pd.RangeIndex(new_index_min, new_index_min + len(new_df))
        else:
            index = non_intersecting_sequence(new_df.index, template.index)

        new_df.index = pd.Index(index, dtype=template.index.dtype)

    return new_df


def get_new_typed_values(
    candidate_values,
    existing_values,
    value_column_label,
    type_column_label,
    candidate_type
):
    '''
    Selects values out of a Series of candidate values if the values are
    either not present in an existing dataframe or they are present but
    do not have the same type as the candidates.

    Parameters
    ----------
    candidate_values : pandas.Series
        Values which may or may not be present in an existing dataframe

    existing_values : pandas.DataFrame
        A dataframe with a column of values to select against and a
        column that identifies the type of each values

    value_column_label
        The label of the column of values in the existing_values
        dataframe

    type_column_label
        The label of the column of types in the existing_values
        dataframe

    candidate_type
        The type of the candidate values

    Returns
    -------
    pandas.Series
        Subset of the candidate values which are either not present in
        the set of existing values or which are present but have a
        different type in the set of existing values
    '''
    candidate_already_exists = existing_values[value_column_label].isin(
        candidate_values
    )
    wrong_type = existing_values[type_column_label] != candidate_type
    candidate_exists_wrong_type = candidate_already_exists & wrong_type

    candidate_is_new_type = candidate_values.isin(
        existing_values.loc[candidate_exists_wrong_type, value_column_label]
    )

    candidate_is_new_value = ~candidate_values.isin(
        existing_values[value_column_label]
    )

    return candidate_values.loc[candidate_is_new_type | candidate_is_new_value]


def missing_integers(input_values, rng=None):
    '''
    Creates a set of integers in a target range that does
    not intersect with values in an input set. Default range
    fills gaps from 0 to the largest input value.

    If target range does not intersect with input, return
    target.

    If target range exactly covers or is a subset of input,
    return empty set.

    input
    -----
    input_values (list-like or set):
        integers to exclude from output

    output
    ------
    (set):
        integers in target range that are not in input set
    '''
    if len(rng) != 2:
        raise ValueError('rng must be list-like with two elements')

    input_values = set(input_values)

    if rng is None:
        rng = [0, int(max(input_values)) + 1]

    target_range = set(range(*rng))

    if target_range.intersection(input_values) == set():
        return target_range
    if target_range.union(input_values) == input_values:
        return set()
    else:
        return set(input_values ^ target_range)


def non_intersecting_sequence(
    to_map,
    existing=[],
    rng=None,
    full=False,
    ignore_duplicates=False
):
    '''
    Maps unique values from a new list-like or set of integers onto the
    smallest sequence of integers that does not intersect with unique
    values in an existing list-like or set of integers.

    inputs
    ------
    to_map (list-like, set, or integer):
        integers to map to integers excluding existing integers. if an
        integer or float with integer value, to_map is converted to a
        range of that length
    existing (list-like or set):
        integers to keep
    rng (integer-valued list-like):
        range of target values for mapping
    full (bool):
        if true, return mapped values appended to existing values
        if false, return only mapped values
    ignore_duplicates (bool):
        if true, map every element in the input set to a unique value in
        the output

    output
    ------
    pandas array:
        existing and mapped integers

    examples
    --------
    a = [2,3,3]
    b = [4,3,4]
    non_intersecting_sequence(a, b)
    # output: Series([0,1,0])

    a = [2,3,3]
    b = [4,3,4]
    non_intersecting_sequence(a, b, rng=[-2,None], full=True)
    # output: Series([4,3,4,-2,-1,-2])

    c = [9,10,9]
    d = [0,2,4]
    non_intersecting_sequence(c, d, ignore_duplicates=True)
    # output: Series([1,3,5])
    '''

    if type(existing) == set:
        existing = list(existing)

    existing = pd.Series(existing, dtype='int')

    if type(to_map) == set:
        to_map = list(to_map)
    elif (type(to_map) == float) and not to_map.is_integer():
        raise ValueError('to_map cannot be a non-integer float')
    elif not iterable_not_string(to_map):
        try:
            int(to_map)
        except TypeError:
            pass
        to_map = range(to_map)

    to_map = pd.Series(to_map, dtype='int')
    unique_ints_to_map = to_map.drop_duplicates()
    len_of_req_sqnce = len(unique_ints_to_map)

    # NOTE
    # I initially had a default value rng=[0, None] in the function
    # signature. In prototyping with Jupyter lab, multiple calls to this
    # function in a row would retain the previous value of rng (the
    # second element would not be None in the second call). I don't know
    # why this happened and I don't know why using rng=None as a default
    # in the signature fixed the problem.

    if rng is None:
        if existing.empty:
            rng = [0, len_of_req_sqnce + 1]
        else:
            rng = [0, int(existing.max()) + len_of_req_sqnce + 1]

    elif len(rng) != 2:
        raise ValueError('rng must be list-like with two elements')

    elif len(range(*rng)) < (len(unique_ints_to_map) + 1):
        raise ValueError(
            'range(*rng) must include at least as many values as '
            'number of values to map'
        )

    available_ints = missing_integers(existing.drop_duplicates(), rng)
    available_ints = pd.Series(list(available_ints), dtype='int')
    available_ints = available_ints.sort_values().reset_index(drop=True)

    if ignore_duplicates:
        new_ints = available_ints[:len(to_map)]
    else:
        new_ints = available_ints.iloc[:len_of_req_sqnce]
        int_map = dict(zip(unique_ints_to_map, new_ints))
        new_ints = to_map.apply(lambda x: int_map[x])

    if full:
        full_sequence = pd.Series(
            existing.append(new_ints, ignore_index=True),
            dtype='int'
        )
        return full_sequence.array
    else:
        return pd.Series(new_ints, dtype='int').array


def replace_escaped_comment_chars(column, comment_char, pattern):

    return column.replace(
        to_replace=pattern,
        value=comment_char,
        regex=True
    )


def set_string_dtype(df):
    '''
    can't use pd.StringDtype() throughout because it currently doesn't
    allow construction with null types other than pd.NA. This will
    likely change soon
    https://github.com/pandas-dev/pandas/pull/41412
    '''
    df_cols = df.columns
    return df.astype(
        dict(
            zip(
                df_cols, [pd.StringDtype()]*len(df_cols)
            )
        )
    )


def strip_csv_comments(column, pattern):

    column = column.str.split(pat=pattern, expand=True)
    return column[0]
