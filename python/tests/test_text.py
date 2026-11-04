"""What memory indexes from text besides the content key, and cutting what goes to Qdrant (text.test.ts and
text-cut.test.ts, ported)."""

from __future__ import annotations

import regex

from memtrails.core.text import children, concept_key, cut, doc_sparse, first_words, query_sparse, term_id, tokens
from memtrails.core.writer import stm_title

#: A UTF-16 unit of a surrogate pair standing alone: not valid UTF-8 once encoded.
LONE = regex.compile("[\ud800-\udfff]")


def test_identifiers_whole_and_by_their_parts() -> None:
    t = tokens("Run deploy.sh on billing_api; see apps/web/main.ts")
    assert {"run", "deploy.sh", "deploy", "sh", "billing_api", "billing", "api", "apps/web/main.ts", "web", "main", "ts"} <= set(t)
    assert "" not in t


def test_other_scripts() -> None:
    assert {"почему", "выгрузка", "functia"} <= set(tokens("Почему не работает выгрузка? Ce face funcția?"))


def test_with_or_without_accents() -> None:
    assert tokens("Bază de date în producție") == tokens("baza de date in productie")
    assert tokens("Ёлка ПРОДАКШН") == tokens("елка продакшн")


def test_scripts_without_spaces() -> None:
    assert {"重建", "容器"} <= set(tokens("我们需要重建网关镜像然后重新创建容器"))
    assert "イメージ" in tokens("ゲートウェイのイメージを再構築する")
    assert "เกตเวย์" in tokens("สร้างอิมเมจเกตเวย์ใหม่")
    assert {"rebuild", "docker", "image"} <= set(tokens("请 rebuild 网关的 docker image"))


def test_devanagari_whole() -> None:
    assert "डेटाबेस" in tokens("डेटाबेस को फिर से बनाओ")


def test_bm25() -> None:
    v = doc_sparse("qdrant qdrant qdrant sidecar")
    assert v["indices"] == sorted(v["indices"])
    assert len(set(v["indices"])) == len(v["indices"])
    q = v["values"][v["indices"].index(term_id("qdrant"))]
    s = v["values"][v["indices"].index(term_id("sidecar"))]
    assert s < q < 3 * s


def test_query_terms_once() -> None:
    q = query_sparse(["memory index", "the memory"])
    assert len(q["indices"]) == len(set(q["indices"]))
    assert all(x == 1 for x in q["values"])
    assert term_id("memory") in q["indices"]


def test_first_words() -> None:
    assert first_words("one  two\nthree four", 3) == "one two three"
    assert first_words("我们需要重建网关镜像", 2) == "我们需要"


def test_children() -> None:
    assert children("one two three") == ["one two three"]
    words = [f"w{i}" for i in range(300)]
    kids = children(" ".join(words))
    assert len(kids) > 2
    assert all(len(k.split(" ")) <= 96 for k in kids)
    assert kids[1].startswith("w72 ")
    assert "w299" in kids[-1]
    body = "我们需要重建网关镜像然后重新创建容器。" * 40
    cjk = children(body)
    assert len(cjk) > 1 and all(len(k) < len(body) for k in cjk)
    many = [f"w{i}" for i in range(5000)]
    seen = {w for k in children(" ".join(many)) for w in k.split(" ")}
    assert all(w in seen for w in many)


def test_concept_keys() -> None:
    assert concept_key(" Proxy.conf ") == "proxy.conf"
    assert concept_key('"Bază   de date",') == "baza de date"
    assert concept_key("acme-proxy:dev") == "acme-proxy:dev"
    assert concept_key("网关 镜像") == "网关 镜像"
    assert concept_key("...") == ""


def test_cut_never_ends_on_half_an_emoji() -> None:
    s = f"{'a' * 99}😀 and more"
    c = cut(s, 100)
    assert not LONE.search(c)
    assert c == f"{'a' * 99}…"
    assert cut(s, 101) == f"{'a' * 99}😀…"
    assert cut("😀", 2) == "😀"
    assert cut("abc", 3) == "abc"
    assert not LONE.search(stm_title({"request": f"{'x' * 99}🎉 party", "title": "thread"}))


def test_a_long_path_gives_its_parts() -> None:
    path = "services/infrastructure/certificates/renewal/hooks/letsencrypt_renewal_hook.sh"
    t = tokens(f"run {path}")
    assert path not in t
    assert {"letsencrypt", "renewal", "hook", "sh", "certificates", "services"} <= set(t)
    assert "a" * 80 not in tokens(f"x/{'a' * 80}/y")
