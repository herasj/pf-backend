import { TweetSchema } from '../../schemas/tweet.schema';
import { TweetsController } from './tweets.controller';
import { MongooseModule } from '@nestjs/mongoose';
import { TweetsService } from './tweets.service';
import { Module } from '@nestjs/common';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: 'tweets', schema: TweetSchema }]),
  ],
  controllers: [TweetsController],
  providers: [TweetsService],
})
export class TweetsModule {}
