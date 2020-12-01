from backend.tf_idf import consolidate_sentences


def test_consolidate():
    first_sentence = "This is the beginning of the sentence,"
    second_sentence = "this is the end of the sentence."

    expected_result = f"{first_sentence} {second_sentence}"

    result = consolidate_sentences(first_sentence, second_sentence)

    assert expected_result == result

