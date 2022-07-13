import pandas as pd
import shorthand as shnd
from bibtexparser.bparser import BibTexParser


def test_parsed_manual_annotation_resolve_links_has_no_nans():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (~parsed.resolve_links().isna()).any().any()



def test_manual_annotation_wrk_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    synthesized = parsed.synthesize_shorthand_entries('wrk', fill_spaces=True)

    check = pd.Series([
        'asmith_bwu__1999__s_bams__101__803__xxx',
        'asmith_bwu__1998__s_bams__100__42__yyy',
        'bjones__1975__s_jats__90__1__!',
        'bwu__1989__t_long|title__!__80__!',
        'Some|Author__1989__t_A|Title|With|\\#__!__!__!',
        'asmith_bwu__2008__s_bams__110__1__zzz'
    ])

    assert (check == synthesized).all()


def test_manual_annotation_note_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    synthesized = parsed.synthesize_shorthand_entries(entry_node_type='note')

    check = pd.Series([
        'this is an article I made up for testing',
        'here\'s a note with an escaped\\__item separator and some "quotation marks"'
    ])

    assert (check == synthesized).all()


def test_single_column_wrk_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/single_column.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=0,
        comment_char='#',
        drop_na=False
    )

    synthesized = parsed.synthesize_shorthand_entries('wrk', fill_spaces=True)

    check = pd.Series([
        'asmith_bwu__1999__s_bams__101__803__xxx',
        'asmith_bwu__1998__s_bams__100__42__yyy',
        'bjones__1975__s_jats__90__1__!',
        'bwu__1989__t_long|title__!__80__!',
        'Some|Author__1989__t_A|Title|With|\\#__!__!__!',
        'asmith_bwu__2008__s_bams__110__1__zzz'
    ])

    assert (check == synthesized).all()


def test_parsed_bibtex_items_resolve_links_has_no_nans():

    bibtex_parser = BibTexParser(common_strings=True)
    bibtex_test_data_fname = "shorthand/test_data/bibtex_test_data_short.bib"
    with open(bibtex_test_data_fname, encoding='utf8') as f:
        bibdatabase = bibtex_parser.parse_file(f)

    data = pd.DataFrame(bibdatabase.entries)

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_bibtex_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_items(
        data.iloc[:4],
        space_char='|',
        na_string_values='!',
        na_node_type='missing'
    )

    parsed.resolve_links().isna().any(axis=None)
