from learn_with_me.backend.video_list_retriever.ted_retriever import _clean_links


def test_get_clean_links_one_link():
    links_href = {"/my_url?language=en"}
    expected_result = {"/my_url"}

    result = _clean_links(links_href)

    assert result == expected_result


def test_get_clean_links_more_than_one_link():
    links_href = {"/my_url?language=en", "/my_url2?language=en"}
    expected_result = {"/my_url", "/my_url2"}

    result = _clean_links(links_href)

    assert result == expected_result


def test_get_clean_links_no_link():
    links_href = {""}
    expected_result = {""}

    result = _clean_links(links_href)

    assert result == expected_result


def test_get_clean_links_nothing_to_be_removed():
    links_href = {"/my_url?language=e", "/my_url2?language=e"}
    expected_result = {"/my_url?language=e", "/my_url2?language=e"}

    result = _clean_links(links_href)

    assert result == expected_result
