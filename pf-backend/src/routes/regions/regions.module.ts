import { RegionsController } from './regions.controller';
import { TweetSchema } from 'src/schemas/tweet.schema';
import { UserSchema } from 'src/schemas/user.schema';
import { RegionsService } from './regions.service';
import { MongooseModule } from '@nestjs/mongoose';
import { Module } from '@nestjs/common';

@Module({
    imports: [ MongooseModule.forFeature([
        { name: 'tweets', schema: TweetSchema },
        { name: 'users', schema: UserSchema },
      ]),],
    controllers: [
        RegionsController, ],
    providers: [
        RegionsService, ],
})
export class RegionsModule {}
