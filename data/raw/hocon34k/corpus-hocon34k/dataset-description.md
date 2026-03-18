# Dataset description


| Column                                            | Name                               |  Description                                                                                                                                |
|:--------------------------------------------------|:-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| newspaper_id                                      | Newspaper ID                       | Pseudonym Identity of the newspaper as consecutive numbers                                                                                  |
| post_id                                           | Post ID                            | Source ID from the newspaper                                                                                                                |
| annotator_id                                      | Annotator ID                       | Pseudonym Identity of the annotators as consecutive numbers                                                                                 |
| phase                                             | Phase                              | 2 or 3                                                                                                                                      |
| split_all                                         | Split for all annotators           | "train", "test", or "val"                                                                                                                   |
| split_12                                          | Split for 12 annotators            | "train", "test", or "val"                                                                                                                   |
| label                                             | Hatespeech and context annotation  | "Hatespeech (enough context)",<br> "Hatespeech (not enough context)",<br> "Not Hatespeech (enough context)" or<br> "Not Hatespeech (not enough context)"|
| label_hs                                          | Hatespeech annotation              | Binary (1 or 0)                                                                                                                             |
| label_context                                     | Context annotation                 | Binary (1 or 0)                                                                                                                             |
| text                                              | Text                               | Text of the comment                                                                                                                         |
