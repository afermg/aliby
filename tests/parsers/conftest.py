#!/usr/bin/env python

from importlib.resources import files
import pytest
import pooch

try:
    examples_dir = files("aliby").parent.parent / "examples" / "logfile_parser"
    grammars_dir = files("logfile_parser") / "grammars"
except Exception:
    examples_dir = None
    grammars_dir = None

REGISTRY = {
    "aggregates_downUpshift_glu_2_0_twice_gcd2_gcd6_gcn3_gcd7_sui2": "md5:c8d141f363152f6f40dc325cb2a79aa2",
    "downUpshift_twice_2_0_2_glu_ura8_ura8h360a_ura8h360r": "md5:5de2bf44b09bb3f5a85cfa125a485f6f",
    "proteinAggregates_starvation_2_0_twice_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "md5:2ca216c295d977cd22b7d7db674f44e6",
    "DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin_secondRound": "md5:425ff7c3387719322d4a5785661b354a",
    "aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_secondRun": "md5:e101a4bc2fd13f8a2125bb667a69c5f3",
    "downUpshift_2_0_2_glu_gcd2_gcd6_gcd7": "md5:2b373a8c8bc99ae7235d2397c76eb204",
    "downUpshift_four_2_0_2_glu_dual_phl__glt1_ura8_ura8": "md5:9882faaf908a517d7751cbe96c7d002d",
    "aggregates_starve_twice_glu_2_0_gcd2_gcd6_gcd7_gcn3_sui2": "md5:7c14ffbe5869fbbaec31375dabbacd97",
    "starve_twice_glu_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "md5:87b59fb902ee7f2512498595c35e77b4",
    "downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__thrice": "md5:2bdc97b5e09df298834bc9bc3984f22b",
    "downUpshift_twice_2_0_2_glu_ura8_phluorinMsn2_phluorinMig1": "md5:934aa9d6d6cd1ee9785aeda2a9620df7",
    "downUpshift_2_0_2_glu_ura8_phl_mig1_phl_msn2": "md5:f445a1320fffedbb8d7ca28b52f6c569",
    "downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__twice": "md5:c28ae615250828688342f30cfc2c23d0",
    "DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin": "md5:58f4501d68fe82cf58537f461e71abb4",
    "starve_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "md5:11fdc38f868164834ceda056b53cc5f6",
    "downUpshift_2_01_2_glucose_dual_pH__dot6_nrg1_tod6": "md5:f7bb797890f45743b58f52502c9288cb",
}

URLS = {
    "aggregates_downUpshift_glu_2_0_twice_gcd2_gcd6_gcn3_gcd7_sui2": "https://zenodo.org/api/records/14187308/files/0_aggregates_downUpshift_glu_2_0_twice_gcd2_gcd6_gcn3_gcd7_sui2log.txt/content",
    "downUpshift_twice_2_0_2_glu_ura8_ura8h360a_ura8h360r": "https://zenodo.org/api/records/14188769/files/0_downUpshift_twice_2_0_2_glu_ura8_ura8h360a_ura8h360rlog.txt/content",
    "proteinAggregates_starvation_2_0_twice_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "https://zenodo.org/api/records/14190257/files/0_proteinAggregates_starvation_2_0_twice_ura7ha_ura7hr_ura8_ura8ha_ura8hrlog.txt/content",
    "DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin_secondRound": "https://zenodo.org/api/records/14188244/files/0_DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin_secondRoundlog.txt/content",
    "aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_secondRun": "https://zenodo.org/api/records/14187963/files/0_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_secondRunlog.txt/content",
    "downUpshift_2_0_2_glu_gcd2_gcd6_gcd7": "https://zenodo.org/api/records/14190058/files/0_downUpshift_2_0_2_glu_gcd2_gcd6_gcd7_log.txt/content",
    "downUpshift_four_2_0_2_glu_dual_phl__glt1_ura8_ura8": "https://zenodo.org/api/records/14189728/files/0_downUpshift_four_2_0_2_glu_dual_phl__glt1_ura8_ura8_log.txt/content",
    "aggregates_starve_twice_glu_2_0_gcd2_gcd6_gcd7_gcn3_sui2": "https://zenodo.org/api/records/14191670/files/0_aggregates_starve_twice_glu_2_0_gcd2_gcd6_gcd7_gcn3_sui2log.txt/content",
    "starve_twice_glu_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "https://zenodo.org/api/records/14187631/files/0_starve_twice_glu_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hrlog.txt/content",
    "downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__thrice": "https://zenodo.org/api/records/14189432/files/0_downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__thricelog.txt/content",
    "downUpshift_twice_2_0_2_glu_ura8_phluorinMsn2_phluorinMig1": "https://zenodo.org/api/records/14189118/files/0_downUpshift_twice_2_0_2_glu_ura8_phluorinMsn2_phluorinMig1log.txt/content",
    "downUpshift_2_0_2_glu_ura8_phl_mig1_phl_msn2": "https://zenodo.org/api/records/14188312/files/0_downUpshift_2_0_2_glu_ura8_phl_mig1_phl_msn2log.txt/content",
    "downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__twice": "https://zenodo.org/api/records/14189505/files/0_downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__twice_log.txt/content",
    "DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin": "https://zenodo.org/api/records/14188123/files/0_DownUpshift_2_0_2_glu_ura_mig1msn2_phluorinlog.txt/content",
    "starve_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hr": "https://zenodo.org/api/records/14191292/files/0_starve_2_0_2_0_ura7ha_ura7hr_ura8_ura8ha_ura8hrlog.txt/content",
    "downUpshift_2_01_2_glucose_dual_pH__dot6_nrg1_tod6": "https://zenodo.org/api/records/14189201/files/0_downUpshift_2_01_2_glucose_dual_pH__dot6_nrg1_tod6_log.txt/content",
}

zenodo_pooch = pooch.create(
    path=pooch.os_cache("aliby"),
    base_url="",
    registry=REGISTRY,
    urls=URLS,
)


@pytest.fixture(scope="module", params=list(REGISTRY.keys()))
def swainlab_log_interface(request) -> str:
    return zenodo_pooch.fetch(request.param)
