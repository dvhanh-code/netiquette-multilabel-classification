# Dataset description


| Column                   | Name                                              |  Description                                                 |
|:-------------------------|:--------------------------------------------------|--------------------------------------------------------------|
| tweet_id                 | Tweet ID                                          | Source ID from Twitter                                       |
| review_text              | Text of the tweet or comment                      | User mentions were replaced by @TwitterUser                  |
| hs                       | Hatespeech annotation                             | Binary (1 or 0)                                              |
| m_hs                     | Misogynistic hatespeech annotation                | Binary (1 or 0)                                              |
| annotation_id            | ID of annotation                                  | Tweets of phase 2 were annotated by all experts              |
| created_at               | Created timestamp of annotation                   |                                                              |
| updated_at               | Updated timestamp of annotation                   |                                                              |
| phase                    | Phase                                             |  1, 2.1, 2.2, 2.3 or 3                                       |
| lead_time                | Elapsed time of annotation                        |                                                              |
| annotator_name           | Annotator name                                    | Pseudonym Identity of the annotators as consecutive numbers  |
| source                   | Source of text                                    | Souce dataset of the text                                    |
| split_hs                 | Source of text                                    | "train", "test", or "val"                                    |
| split_m_hs               | Source of text                                    | "train", "test", or "val"                                    |
