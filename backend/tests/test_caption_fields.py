"""The structured tail of a video caption is parsed, never trusted (C12-C).

A formatting slip by the model must cost the FIELDS, never the caption: the
whole text stays the prose and fields come back None. Stdlib-only module — it
is imported by path from the inference interpreter."""
from app.services.caption_fields import (
    FIELD_KEYS, fields_to_prose, split_caption_fields,
)

PARA = ('A woman in a dark sweater walks toward the window, her hair swinging as she '
        'turns, and the light catches her face when she stops.')


def test_a_well_formed_tail_yields_prose_and_five_fields():
    raw = (PARA + '\n---\nSubject: a woman in a dark sweater\nMotion: walks to the window '
           'and turns\nSetting: a bright apartment\nStyle: soft daylight, handheld feel\n'
           'Short: a woman walks to a bright window and turns into the light')
    prose, fields = split_caption_fields(raw)
    assert prose == PARA
    assert tuple(fields) == FIELD_KEYS
    assert fields['motion'] == 'walks to the window and turns'
    assert fields['short'].startswith('a woman walks')


def test_markdown_bold_labels_and_missing_keys_are_tolerated():
    raw = PARA + '\n\n---\n\n**Subject:** a woman\n**Motion:** she turns\n'
    prose, fields = split_caption_fields(raw)
    assert prose == PARA
    assert fields['subject'] == 'a woman' and fields['motion'] == 'she turns'
    assert fields['setting'] is None and fields['short'] is None


def test_no_separator_but_a_subject_line_still_splits():
    raw = PARA + '\nSubject: a woman\nMotion: turns'
    prose, fields = split_caption_fields(raw)
    assert prose == PARA
    assert fields['subject'] == 'a woman'


def test_a_caption_without_fields_is_kept_whole():
    prose, fields = split_caption_fields(PARA)
    assert prose == PARA and fields is None
    # A separator followed by nothing labelled: the dash line is not a caption.
    prose2, fields2 = split_caption_fields(PARA + '\n---\nnothing labelled here')
    assert fields2 is None
    assert prose2.startswith(PARA)


def test_empty_input_is_empty_output_never_an_error():
    assert split_caption_fields('') == ('', None)
    assert split_caption_fields(None) == ('', None)


def test_fields_compose_into_short_prose_in_reading_order():
    text = fields_to_prose({'subject': 'a woman in red', 'motion': 'walks left',
                            'setting': 'a quiet street', 'style': 'golden hour.',
                            'short': 'ignored here'})
    assert text == 'a woman in red. walks left. a quiet street. golden hour.'
    assert fields_to_prose(None) == ''
    assert fields_to_prose({'subject': None, 'motion': ''}) == ''
