-- 0072_collection_titles.sql
-- Rename collection display titles to the curated, consistently-formatted names
-- (surname-first for person files; expanded acronyms; source archive noted).
-- Display-only change (collections.title); slugs, ids, and all data are untouched.
-- Reversible: prior titles are recorded in the comment block below.
--
-- Prior titles (for rollback):
--   venona                        = 'Venona Decrypts'
--   vassiliev                     = 'Vassiliev Notebooks'
--   huac_hearings                 = 'HUAC Hearings (1947-1957)'
--   huac_reports                  = 'HUAC Reports (1948-1964)'
--   rosenberg                     = 'Julius Rosenberg FBI Case Files'
--   fbicomrap                     = 'FBI COMRAP (Comintern Apparatus) Files'
--   solo                          = 'FBI SOLO Operation Files'
--   mccarthy                      = 'McCarthy Hearings (1953-1954)'
--   silvermaster                  = 'FBI Silvermaster files'
--   siss_scope_soviet             = 'Scope of Soviet Activity in the United States (SISS, 1956- )'
--   fbi_cinrad                    = 'FBI CINRAD (Communist Infiltration of the Radiation Laboratory) Summary'
--   soviet_atomic_espionage_1951  = 'Soviet Atomic Espionage (Joint Committee on Atomic Energy, 1951)'
--   fbi_hiskey                    = 'FBI Files on Clarence Hiskey'
--   david_greenglass              = 'David Greenglass FBI Files'
--   david_ruth_greenglass         = 'David and Ruth Greenglass FBI Files'
--   golos                         = 'FBI Files on Jacob Golos'
--   pravdin                       = 'MI5 Security Service Files on Vladimir Pravdin (TNA KV 2/1898)'
--   soviet_intel_travel_techniques= 'Soviet Intelligence Travel and Intelligence Techniques (CIA)'
--   hiss_chambers                 = 'FBI Files on Alger Hiss & Whittaker Chambers'
--   winton_burdett                = 'FBI Files on Winton Burdett'
--   judith_coplon                 = 'FBI Files on Judith Coplon'
--   jack_childs                   = 'FBI Files on Jack Childs'
--   morris_childs                 = 'FBI Files on Morris Childs (Operation SOLO)'
--   volodarsky                    = 'MI5 Files (Volodarsky / Feldman, TNA KV 2/2881-2882)'
--   mink                          = 'MI5 File on George Mink (TNA KV 2/2067)'
--   albertson                     = 'FBI File on William Albertson (HQ 65-38100)'
--   eva_childs                    = 'FBI Files on Eva Childs'
-- (brothman_moskowitz_grand_jury, rosenberg_grand_jury, rosenberg_trial_transcripts,
--  oscar_seborer already match the target names and are left unchanged.)

UPDATE collections SET title = 'Venona Decryptions, National Security Agency'                                                                     WHERE slug = 'venona';
UPDATE collections SET title = 'Vassiliev, Alexander Notebooks'                                                                                   WHERE slug = 'vassiliev';
UPDATE collections SET title = 'HUAC (House Un-American Activities Committee) Hearings (1947-1957)'                                                WHERE slug = 'huac_hearings';
UPDATE collections SET title = 'HUAC (House Un-American Activities Committee) Reports (1948-1964)'                                                 WHERE slug = 'huac_reports';
UPDATE collections SET title = 'Rosenberg, Julius FBI Files'                                                                                      WHERE slug = 'rosenberg';
UPDATE collections SET title = 'COMRAP (Comintern Apparatus) FBI Files'                                                                           WHERE slug = 'fbicomrap';
UPDATE collections SET title = 'SOLO Operation FBI Files'                                                                                         WHERE slug = 'solo';
UPDATE collections SET title = 'McCarthy Senate Subcommittee Hearings (1953-1954)'                                                                WHERE slug = 'mccarthy';
UPDATE collections SET title = 'Silvermaster, Nathan FBI files'                                                                                   WHERE slug = 'silvermaster';
UPDATE collections SET title = 'Scope of Soviet Activity in the United States hearings and reports, Senate Internal Security Subcommittee (1956-1959)' WHERE slug = 'siss_scope_soviet';
UPDATE collections SET title = 'CINRAD (Communist Infiltration of the Radiation Laboratory) FBI Summary'                                          WHERE slug = 'fbi_cinrad';
UPDATE collections SET title = 'Soviet Atomic Espionage hearings and reports, Congressional Joint Committee on Atomic Energy (1951)'               WHERE slug = 'soviet_atomic_espionage_1951';
UPDATE collections SET title = 'Hiskey, Clarence FBI Files'                                                                                       WHERE slug = 'fbi_hiskey';
UPDATE collections SET title = 'Greenglass, David FBI Files'                                                                                      WHERE slug = 'david_greenglass';
UPDATE collections SET title = 'Greenglass, David and Ruth FBI Files'                                                                             WHERE slug = 'david_ruth_greenglass';
UPDATE collections SET title = 'Golos, Jacob FBI files'                                                                                           WHERE slug = 'golos';
UPDATE collections SET title = 'Pravdin, Vladimir British Security Service-MI5 files (KV 2/1898)'                                                  WHERE slug = 'pravdin';
UPDATE collections SET title = 'Soviet Intelligence Travel and Intelligence Techniques, CIA report'                                               WHERE slug = 'soviet_intel_travel_techniques';
UPDATE collections SET title = 'Hiss, Alger and Whittaker Chambers FBI files'                                                                     WHERE slug = 'hiss_chambers';
UPDATE collections SET title = 'Burdett, Winton FBI files'                                                                                        WHERE slug = 'winton_burdett';
UPDATE collections SET title = 'Coplon, Judith FBI files'                                                                                         WHERE slug = 'judith_coplon';
UPDATE collections SET title = 'Childs, Jack (Operation SOLO) FBI files'                                                                          WHERE slug = 'jack_childs';
UPDATE collections SET title = 'Childs, Morris (Operation SOLO) FBI files'                                                                        WHERE slug = 'morris_childs';
UPDATE collections SET title = 'Volodarsky, Iosif (aka Armand Feldman) British Security Service-MI5 files (KV 2/2881–2882)'                        WHERE slug = 'volodarsky';
UPDATE collections SET title = 'Mink, George British Security Service-MI5 files (KV 2/2067)'                                                       WHERE slug = 'mink';
UPDATE collections SET title = 'Albertson, William FBI files (HQ 65-38100)'                                                                       WHERE slug = 'albertson';
UPDATE collections SET title = 'Childs, Eva (Operation SOLO) FBI Files'                                                                           WHERE slug = 'eva_childs';
