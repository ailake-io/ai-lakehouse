import json
import tempfile
import unittest
from pathlib import Path

from scripts.plugin_release import resolve
from scripts.python_release import git
from scripts.release_audit import check_assets, latest_version


class PluginReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        self.source = self.root / 'VERSION'
        self.docs = self.root / 'README.md'
        self.source.write_text('1.2.3\n')
        self.docs.write_text('demo-v1.2.3 and demo-v1.2.3\n')
        self.plugin = {'id':'demo', 'version_policy':'independent', 'tag_prefix':'demo-v',
                       'targets':[{'file':'VERSION','version_pattern':r'^([0-9.]+)'},
                                  {'file':'README.md','version_pattern':r'demo-v([0-9.]+)','multiple':True}]}
        (self.root / 'scripts/plugins.json').write_text(json.dumps({'plugins':[self.plugin]}))
        git(self.root, 'init', '-b', 'main')
        git(self.root, 'config', 'user.name', 'Release tests')
        git(self.root, 'config', 'user.email', 'release-tests@example.invalid')
        self.commit()

    def commit(self):
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-m', 'Plugin fixture')

    def test_blank_input_uses_prepared_version_without_incrementing(self):
        result = resolve(self.root,'demo','','refs/heads/main')
        self.assertEqual(result,{'version':'1.2.3','tag':'demo-v1.2.3',
                                 'source_sha':git(self.root,'rev-parse','HEAD')})

    def test_new_version_must_be_prepared_in_source_and_docs(self):
        with self.assertRaisesRegex(ValueError,'Prepare this version'):
            resolve(self.root,'demo','1.2.4','refs/heads/main')
        self.docs.write_text('demo-v1.2.3 and demo-v1.2.4\n')
        with self.assertRaisesRegex(ValueError,'documentation versions differ'):
            resolve(self.root,'demo','','refs/heads/main')

    def test_retry_preserves_existing_tag_after_main_advances(self):
        git(self.root,'tag','demo-v1.2.3')
        original=git(self.root,'rev-parse','HEAD')
        self.source.write_text('1.2.4\n')
        self.docs.write_text('demo-v1.2.4\n')
        self.commit()
        self.assertEqual(resolve(self.root,'demo','1.2.3','refs/heads/main')['source_sha'],original)

    def test_rejects_tag_with_wrong_source_version(self):
        git(self.root,'tag','demo-v1.2.4')
        with self.assertRaisesRegex(ValueError,'inconsistent source metadata'):
            resolve(self.root,'demo','1.2.4','refs/heads/main')

    def test_refuses_arbitrary_branches(self):
        with self.assertRaisesRegex(ValueError,'must run from main'):
            resolve(self.root,'demo','','refs/heads/develop')


class PublishedAssetTests(unittest.TestCase):
    def setUp(self):
        self.plugin={'id':'demo','tag_prefix':'demo-v',
                     'artifacts':['demo-{version}.jar','native-linux.so','native-windows.dll']}
        self.release={'tag_name':'demo-v1.2.3','assets':[{'name':name} for name in
                      ('demo-1.2.3.jar','native-linux.so','native-windows.dll')]}

    def test_latest_version_uses_semver_and_ignores_other_components(self):
        releases = [{'tag_name':tag} for tag in ('demo-v1.2.9','demo-v1.2.10','other-v9.0.0')]
        releases += [{'tag_name':'demo-v1.3.0','draft':True},
                     {'tag_name':'demo-v2.0.0','prerelease':True}]
        self.assertEqual(latest_version(self.plugin,releases),'1.2.10')

    def test_complete_release_matches_version_and_platforms(self):
        self.assertEqual(check_assets(self.plugin,'1.2.3',self.release),[])

    def test_missing_platform_or_wrong_version_is_detected(self):
        self.release['assets'].pop()
        self.assertTrue(any('native-windows.dll' in e for e in check_assets(self.plugin,'1.2.3',self.release)))
        self.assertTrue(any('demo-1.2.4.jar' in e for e in check_assets(self.plugin,'1.2.4',self.release)))

    def test_draft_or_prerelease_is_not_a_stable_publication(self):
        for flag in ('draft','prerelease'):
            with self.subTest(flag=flag):
                self.release[flag]=True
                self.assertTrue(check_assets(self.plugin,'1.2.3',self.release))
                self.release[flag]=False


if __name__ == '__main__':
    unittest.main()
