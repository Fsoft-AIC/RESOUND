from .transformer import *




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# texts = ["WE HAVE A RESEARCH IDEA THAT WE THINK IS PERFECT FOR YOU IN OTHER WORDS HE DIDN'T HAVE THE EVIDENCE WE WORK VERY SLOWLY FOR ALL OF YOU WHO HAVE BEEN THROUGH HUNDREDS OF CASES BUT IN EVERY SINGLE INSTANCE YOU HAD MADE SOME NOT WHAT HE DID SO HOW DOES YOUR BRAIN GIVE YOU THAT DETAIL", "WHY AREN'T THESE PEOPLE QUESTIONING US", 'HOW DO YOU FIT INTO DINOSAUR', "THAT'S THE PASSION OF THE GEOLOGICAL RECORD", "AND I'M NOT A HUGE PROBLEM", "SO WHEN YOU'RE BORED YOU CAN MAKE FEELINGS LIKE COMMENTS AND", 'THIS IS MY STORY', 'SO THAT WAS IT', 'WE HAVE DONE THE RESEARCH', 'WHEN WORK IS GOING TO DO THE REAL THING']

# texts = [process_text(text) for text in texts]
# texts = [np.array(text) for text in texts]
# text_lens = np.array([text.shape[0] for text in texts])
# text_lens = torch.from_numpy(text_lens).long().to(device)
# texts, padding_mask = pad_1D(texts)
# texts = torch.from_numpy(texts).long().to(device)
# text_mask = get_mask_from_lengths(text_lens, device=device)

# encoder = Encoder(device=device)
# encoder_output = encoder(texts, text_mask)
